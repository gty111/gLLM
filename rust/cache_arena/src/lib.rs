//! CPU-only aligned cache ownership. No tensor/device operations or unsafe code.
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

type Extent = (usize, usize);
struct Group {
    span: usize,
    high: bool,
    occupied: Vec<usize>,
    live: Vec<bool>,
    free: Vec<bool>,
    retained: Vec<bool>,
    priority: Vec<u64>,
    cold_bits: Vec<u64>,
    next_word: usize,
    heap: BinaryHeap<Reverse<(u64, i64)>>,
    free_count: usize,
    live_count: usize,
}
impl Group {
    fn key(&self, slot: usize) -> i64 {
        if self.high {
            -(slot as i64)
        } else {
            slot as i64
        }
    }
    fn extent(&self, slot: usize) -> PyResult<Extent> {
        if slot >= self.live.len() {
            return Err(PyIndexError::new_err("arena slot out of range"));
        }
        Ok((slot * self.span, (slot + 1) * self.span))
    }
    fn overlapping(&self, start: usize, end: usize) -> std::ops::Range<usize> {
        start / self.span..((end - 1) / self.span + 1).min(self.live.len())
    }
    fn cold_free(&mut self, slot: usize) {
        let word = slot / 64;
        self.cold_bits[word] |= 1u64 << (slot % 64);
        self.next_word = if self.high {
            self.next_word.max(word)
        } else {
            self.next_word.min(word)
        };
    }
    fn build_heap(&mut self) {
        self.heap = BinaryHeap::from(
            (0..self.live.len())
                .filter(|&s| self.free[s])
                .map(|s| Reverse((self.priority[s], self.key(s))))
                .collect::<Vec<_>>(),
        );
    }
    fn queue(&mut self, slot: usize, cold: bool) {
        if cold {
            self.cold_free(slot);
            return;
        }
        self.heap
            .push(Reverse((self.priority[slot], self.key(slot))));
        if self.heap.len() > 4 * self.live.len() + 64 {
            self.build_heap();
        }
    }
    fn queue_batch(&mut self, slots: Vec<usize>, cold: bool) {
        if cold {
            for slot in slots {
                self.cold_free(slot);
            }
            return;
        }
        let entries: Vec<_> = slots
            .into_iter()
            .map(|s| Reverse((self.priority[s], self.key(s))))
            .collect();
        self.heap.extend(entries);
        if self.heap.len() > 4 * self.live.len() + 64 {
            self.build_heap();
        }
    }
    fn take(&mut self, cold: bool) -> PyResult<usize> {
        if cold {
            while self.next_word < self.cold_bits.len() {
                let bits = self.cold_bits[self.next_word];
                if bits != 0 {
                    let bit = if self.high {
                        63 - bits.leading_zeros()
                    } else {
                        bits.trailing_zeros()
                    };
                    let slot = self.next_word * 64 + bit as usize;
                    self.cold_bits[self.next_word] &= !(1u64 << bit);
                    self.free[slot] = false;
                    return Ok(slot);
                }
                if self.high {
                    if self.next_word == 0 {
                        break;
                    }
                    self.next_word -= 1;
                } else {
                    self.next_word += 1;
                }
            }
        } else {
            while let Some(Reverse((priority, key))) = self.heap.pop() {
                let slot = if self.high { -key } else { key } as usize;
                if self.free[slot] && priority == self.priority[slot] {
                    self.free[slot] = false;
                    return Ok(slot);
                }
            }
        }
        Err(PyRuntimeError::new_err(
            "arena free-slot index is inconsistent",
        ))
    }
}
fn merge(mut ranges: Vec<Extent>) -> Vec<Extent> {
    ranges.sort_unstable();
    let mut result: Vec<Extent> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = result.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        result.push((start, end));
    }
    result
}
#[pyclass]
struct ArenaCore {
    pages: usize,
    // One byte per physical page; Python never maintains a shadow owner map.
    owned: Vec<u8>,
    // Materialize ages only when a prefix is actually retained.
    ages: Vec<u64>,
    clock: u64,
    used: usize,
    groups: Vec<Group>,
    changed: Vec<Extent>,
    claimed: Vec<usize>,
}
impl ArenaCore {
    fn check_group(&self, id: usize) -> PyResult<()> {
        if id >= self.groups.len() {
            Err(PyIndexError::new_err("unknown cache type"))
        } else {
            Ok(())
        }
    }
    fn update(&mut self, start: usize, end: usize, claim: bool) {
        let cold = self.clock == 0;
        for g in &mut self.groups {
            let overlapping = g.overlapping(start, end);
            let bulk = overlapping.len() > 32;
            let mut newly_free = Vec::new();
            for s in overlapping {
                let a = s * g.span;
                let b = a + g.span;
                let overlap = end.min(b) - start.max(a);
                let old = g.occupied[s];
                let new = if claim { old + overlap } else { old - overlap };
                debug_assert!(new <= g.span);
                g.occupied[s] = new;
                g.free[s] = new == 0;
                if cold && new != 0 {
                    g.cold_bits[s / 64] &= !(1u64 << (s % 64));
                }
                if old == 0 && new != 0 {
                    g.free_count -= 1;
                }
                if new == 0 && old != 0 {
                    g.free_count += 1;
                    g.priority[s] = if cold {
                        0
                    } else {
                        *self.ages[a..b].iter().max().unwrap()
                    };
                    if bulk {
                        newly_free.push(s);
                    } else {
                        g.queue(s, cold);
                    }
                }
            }
            if !newly_free.is_empty() {
                g.queue_batch(newly_free, cold);
            }
        }
    }
    fn refresh_ages(&mut self, ranges: Vec<Extent>) {
        for (start, end) in merge(ranges) {
            for g in &mut self.groups {
                for s in g.overlapping(start, end) {
                    if g.occupied[s] != 0 {
                        continue;
                    }
                    let a = s * g.span;
                    let priority = *self.ages[a..a + g.span].iter().max().unwrap();
                    if priority != g.priority[s] {
                        g.priority[s] = priority;
                        g.queue(s, false);
                    }
                }
            }
        }
    }
    fn claim_slots(&mut self, id: usize, slots: &[usize]) {
        self.claimed = slots.to_vec();
        let g = &mut self.groups[id];
        let mut ranges = Vec::with_capacity(slots.len());
        for &s in slots {
            let a = s * g.span;
            let b = a + g.span;
            g.live[s] = true;
            g.live_count += 1;
            self.owned[a..b].fill(1);
            self.used += g.span;
            ranges.push((a, b));
        }
        self.changed = merge(ranges);
        for i in 0..self.changed.len() {
            let (a, b) = self.changed[i];
            self.update(a, b, true);
        }
    }
}
#[pymethods]
impl ArenaCore {
    #[new]
    fn new(pages: usize) -> PyResult<Self> {
        if pages == 0 {
            return Err(PyValueError::new_err("arena must have physical pages"));
        }
        Ok(Self {
            pages,
            owned: vec![0; pages],
            ages: Vec::new(),
            clock: 0,
            used: 0,
            groups: Vec::new(),
            changed: Vec::new(),
            claimed: Vec::new(),
        })
    }
    fn register(&mut self, span: usize, high: bool) -> PyResult<usize> {
        if span == 0 {
            return Err(PyValueError::new_err("slot span must be positive"));
        }
        let n = self.pages / span;
        let mut g = Group {
            span,
            high,
            occupied: vec![0; n],
            live: vec![false; n],
            free: vec![false; n],
            retained: vec![false; n],
            priority: vec![0; n],
            cold_bits: vec![0; n.div_ceil(64)],
            next_word: if high {
                n.div_ceil(64).saturating_sub(1)
            } else {
                0
            },
            heap: BinaryHeap::new(),
            free_count: 0,
            live_count: 0,
        };
        let mut entries = Vec::with_capacity(n);
        for s in 0..n {
            let a = s * span;
            g.occupied[s] = if self.used == 0 {
                0
            } else {
                self.owned[a..a + span].iter().map(|&x| x as usize).sum()
            };
            g.priority[s] = if self.clock == 0 {
                0
            } else {
                *self.ages[a..a + span].iter().max().unwrap()
            };
            if g.occupied[s] == 0 {
                g.free_count += 1;
                g.free[s] = true;
                if self.clock == 0 {
                    g.cold_free(s);
                } else {
                    entries.push(Reverse((g.priority[s], g.key(s))));
                }
            }
        }
        g.heap = BinaryHeap::from(entries);
        self.groups.push(g);
        Ok(self.groups.len() - 1)
    }
    #[pyo3(signature=(id, count, slot=None, retain=false))]
    fn allocate(
        &mut self,
        id: usize,
        count: usize,
        slot: Option<usize>,
        retain: bool,
    ) -> PyResult<Option<Vec<usize>>> {
        self.check_group(id)?;
        self.changed.clear();
        self.claimed.clear();
        if (slot.is_some() && count != 1) || (retain && slot.is_none()) {
            return Err(PyValueError::new_err("invalid exact-slot allocation"));
        }
        if count == 0 {
            return Ok(Some(Vec::new()));
        }
        let g = &mut self.groups[id];
        let slots = if let Some(s) = slot {
            g.extent(s)?;
            if g.live[s] {
                return if retain {
                    Ok(Some(vec![s]))
                } else {
                    Err(PyRuntimeError::new_err("arena slot is already live"))
                };
            }
            if g.occupied[s] != 0 {
                return Ok(None);
            }
            vec![s]
        } else {
            if g.free_count < count {
                return Ok(None);
            }
            let mut selected = Vec::with_capacity(count);
            for _ in 0..count {
                selected.push(g.take(self.clock == 0)?);
            }
            selected
        };
        self.claim_slots(id, &slots);
        Ok(Some(slots))
    }
    fn retain_many(&mut self, id: usize, slots: Vec<usize>) -> PyResult<Option<Vec<usize>>> {
        self.check_group(id)?;
        let g = &self.groups[id];
        let mut seen = HashSet::with_capacity(slots.len());
        let mut unique = Vec::with_capacity(slots.len());
        let mut fresh = Vec::new();
        for s in slots {
            g.extent(s)?;
            if !seen.insert(s) {
                continue;
            }
            if !g.live[s] && g.occupied[s] != 0 {
                return Ok(None);
            }
            unique.push(s);
            if !g.live[s] {
                fresh.push(s);
            }
        }
        self.claim_slots(id, &fresh);
        Ok(Some(unique))
    }
    fn free(&mut self, id: usize, slots: Vec<usize>) -> PyResult<()> {
        self.check_group(id)?;
        let mut sorted = slots;
        sorted.sort_unstable();
        let g = &self.groups[id];
        for (i, &s) in sorted.iter().enumerate() {
            g.extent(s)?;
            if (i > 0 && sorted[i - 1] == s) || !g.live[s] {
                return Err(PyRuntimeError::new_err(
                    "duplicate or unowned arena slot in free",
                ));
            }
        }
        let g = &mut self.groups[id];
        let mut ranges = Vec::with_capacity(sorted.len());
        for s in sorted {
            let a = s * g.span;
            let b = a + g.span;
            g.live[s] = false;
            g.live_count -= 1;
            self.owned[a..b].fill(0);
            self.used -= g.span;
            ranges.push((a, b));
        }
        self.changed = merge(ranges);
        for i in 0..self.changed.len() {
            let (a, b) = self.changed[i];
            self.update(a, b, false);
        }
        Ok(())
    }
    fn mark_cached(&mut self, id: usize, slots: Vec<usize>) -> PyResult<()> {
        self.check_group(id)?;
        let g = &self.groups[id];
        for &s in &slots {
            g.extent(s)?;
            if !g.live[s] {
                return Err(PyRuntimeError::new_err("cannot cache unowned arena slot"));
            }
        }
        if !slots.is_empty() && self.ages.is_empty() {
            self.ages = vec![0; self.pages];
            // Prefix ages switch the same allocator from a cold ordered bitset
            // to age-aware heaps. The bitsets are no longer needed afterwards.
            for group in &mut self.groups {
                group.build_heap();
                group.cold_bits = Vec::new();
            }
        }
        for s in slots {
            let g = &mut self.groups[id];
            self.clock = self
                .clock
                .checked_add(1)
                .ok_or_else(|| PyRuntimeError::new_err("cache age overflow"))?;
            self.ages[s * g.span..(s + 1) * g.span].fill(self.clock);
            g.retained[s] = true;
        }
        Ok(())
    }
    fn discard_cached(&mut self, id: usize, slots: Vec<usize>) -> PyResult<()> {
        self.check_group(id)?;
        for &s in &slots {
            self.groups[id].extent(s)?;
        }
        let g = &mut self.groups[id];
        let mut ranges = Vec::new();
        for s in slots {
            if g.retained[s] {
                g.retained[s] = false;
                let a = s * g.span;
                let b = a + g.span;
                self.ages[a..b].fill(0);
                ranges.push((a, b));
            }
        }
        self.refresh_ages(ranges);
        Ok(())
    }
    // Determine stale views while the new owner is already pinned. Python
    // invokes policy callbacks after this method returns (no Rust borrow held).
    fn invalidate(
        &mut self,
        id: usize,
        slots: Vec<usize>,
        callbacks: Vec<bool>,
        preserve: bool,
    ) -> PyResult<Vec<(usize, usize)>> {
        self.check_group(id)?;
        let span = self.groups[id].span;
        for &s in &slots {
            self.groups[id].extent(s)?;
        }
        let mut seen = HashSet::new();
        let mut notify = Vec::new();
        let mut invalidated = Vec::new();
        let protected: HashSet<_> = if preserve {
            slots.iter().copied().collect()
        } else {
            HashSet::new()
        };
        for s in slots {
            let a = s * span;
            let b = a + span;
            for (i, g) in self.groups.iter_mut().enumerate() {
                let callback = callbacks.get(i).copied().unwrap_or(false);
                for t in g.overlapping(a, b) {
                    if i == id && protected.contains(&t) {
                        continue;
                    }
                    if !seen.insert((i, t)) {
                        continue;
                    }
                    if g.retained[t] {
                        g.retained[t] = false;
                        let x = t * g.span;
                        let y = x + g.span;
                        self.ages[x..y].fill(0);
                        invalidated.push((x, y));
                    }
                    if callback {
                        notify.push((i, t));
                    }
                }
            }
        }
        self.refresh_ages(invalidated);
        Ok(notify)
    }
    fn counts(&self, id: usize) -> PyResult<(usize, usize)> {
        self.check_group(id)?;
        let g = &self.groups[id];
        Ok((g.free_count, g.live_count))
    }
    fn status(&self) -> (usize, u64) {
        (self.used, self.clock)
    }
    fn is_free(&self, id: usize, s: usize) -> PyResult<bool> {
        self.check_group(id)?;
        let g = &self.groups[id];
        g.extent(s)?;
        Ok(g.occupied[s] == 0)
    }
    fn slots(&self, id: usize, kind: &str) -> PyResult<Vec<usize>> {
        self.check_group(id)?;
        let g = &self.groups[id];
        Ok((0..g.live.len())
            .filter(|&s| match kind {
                "live" => g.live[s],
                "retained" => g.retained[s],
                _ => g.occupied[s] == 0,
            })
            .collect())
    }
    fn occupancy(&self, id: usize) -> PyResult<Vec<usize>> {
        self.check_group(id)?;
        Ok(self.groups[id].occupied.clone())
    }
    fn priorities(&self, id: usize) -> PyResult<Vec<u64>> {
        self.check_group(id)?;
        Ok(self.groups[id].priority.clone())
    }
    fn heap(&self, id: usize) -> PyResult<Vec<(u64, i64)>> {
        self.check_group(id)?;
        Ok(self.groups[id].heap.iter().map(|x| x.0).collect())
    }
    fn ages(&self) -> Vec<u64> {
        if self.ages.is_empty() {
            vec![0; self.pages]
        } else {
            self.ages.clone()
        }
    }
    fn ranges(&self) -> Vec<(usize, usize, usize, usize)> {
        let mut result = Vec::new();
        for (i, g) in self.groups.iter().enumerate() {
            for (s, &live) in g.live.iter().enumerate() {
                if live {
                    result.push((i, s, s * g.span, (s + 1) * g.span));
                }
            }
        }
        result
    }
    fn changed_ranges(&self) -> Vec<Extent> {
        self.changed.clone()
    }
    fn claimed_slots(&self) -> Vec<usize> {
        self.claimed.clone()
    }
}
#[pymodule]
fn _cache_arena(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArenaCore>()?;
    Ok(())
}
