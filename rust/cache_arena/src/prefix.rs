//! Prefix metadata only. Tensor writes and arena ownership remain with their owners.
use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use std::collections::HashMap;

#[derive(Default)]
struct Page {
    hash: Option<isize>,
    canary: Option<Vec<i64>>,
    refs: usize,
    snapshot: Option<usize>,
    valid: bool,
    pins: usize,
    older: Option<usize>,
    newer: Option<usize>,
    queued: bool,
}
#[pyclass(module = "gllm._cache_arena")]
pub struct PrefixCore {
    pages: Vec<Page>,
    index: HashMap<isize, usize>,
    oldest: Option<usize>,
    newest: Option<usize>,
    reclaimable_count: usize,
}
impl PrefixCore {
    fn page(&self, p: usize) -> PyResult<&Page> {
        self.pages
            .get(p)
            .ok_or_else(|| PyIndexError::new_err("prefix page out of range"))
    }
    fn untouch(&mut self, p: usize) {
        if !self.pages[p].queued {
            return;
        }
        let (older, newer) = (self.pages[p].older, self.pages[p].newer);
        if let Some(i) = older {
            self.pages[i].newer = newer;
        } else {
            self.oldest = newer;
        }
        if let Some(i) = newer {
            self.pages[i].older = older;
        } else {
            self.newest = older;
        }
        self.pages[p].older = None;
        self.pages[p].newer = None;
        self.pages[p].queued = false;
        self.reclaimable_count -= 1;
    }
    fn touch(&mut self, p: usize) {
        self.untouch(p);
        if self.pages[p].valid && self.pages[p].pins == 0 {
            self.pages[p].older = self.newest;
            if let Some(i) = self.newest {
                self.pages[i].newer = Some(p);
            } else {
                self.oldest = Some(p);
            }
            self.newest = Some(p);
            self.pages[p].queued = true;
            self.reclaimable_count += 1;
        }
    }
}
#[pymethods]
impl PrefixCore {
    #[new]
    fn new(count: usize) -> Self {
        Self {
            pages: (0..count).map(|_| Page::default()).collect(),
            index: HashMap::new(),
            oldest: None,
            newest: None,
            reclaimable_count: 0,
        }
    }
    fn lookup(&self, hash: isize, canary: Vec<i64>) -> Option<usize> {
        self.index
            .get(&hash)
            .copied()
            .filter(|&p| self.pages[p].canary.as_ref() == Some(&canary))
    }
    fn scan(
        &self,
        src: &Bound<'_, PyList>,
        hashes: &mut PrefixHashes,
        page_size: usize,
        count: usize,
        canary: Vec<i64>,
    ) -> PyResult<Vec<usize>> {
        if page_size == 0 || count > src.len() / page_size {
            return Err(PyRuntimeError::new_err("invalid prefix scan length"));
        }
        let mut result = Vec::new();
        for i in 0..count {
            let hash = hashes.page_hash(src, page_size, i)?;
            match self.index.get(&hash).copied() {
                Some(p) if self.pages[p].canary.as_ref() == Some(&canary) => result.push(p),
                _ => break,
            }
        }
        Ok(result)
    }
    fn register(&mut self, p: usize, hash: isize, canary: Vec<i64>, replace: bool) -> PyResult<()> {
        self.page(p)?;
        if !replace && self.index.contains_key(&hash) {
            return Ok(());
        }
        if let Some(old) = self.pages[p].hash {
            if self.index.get(&old) == Some(&p) {
                self.index.remove(&old);
            }
        }
        self.pages[p].hash = Some(hash);
        self.pages[p].canary = Some(canary);
        self.index.insert(hash, p);
        Ok(())
    }
    fn retain(&mut self, pages: Vec<usize>) -> PyResult<()> {
        for &p in &pages {
            self.page(p)?;
        }
        for p in pages {
            self.pages[p].refs += 1;
        }
        Ok(())
    }
    fn release(&mut self, pages: Vec<usize>) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        // Validate the entire batch before changing any reference counts.
        let mut sorted = pages.clone();
        sorted.sort_unstable();
        let mut i = 0;
        while i < sorted.len() {
            let p = sorted[i];
            let start = i;
            while i < sorted.len() && sorted[i] == p {
                i += 1;
            }
            if self.page(p)?.refs < i - start {
                return Err(PyRuntimeError::new_err("prefix reference underflow"));
            }
        }
        let mut released = Vec::new();
        for p in pages {
            self.pages[p].refs -= 1;
            if self.pages[p].refs == 0 {
                released.push(p);
            }
        }
        let mut cached = Vec::new();
        let mut discarded = Vec::new();
        for &p in released.iter().rev() {
            if self.pages[p]
                .hash
                .is_some_and(|h| self.index.get(&h) == Some(&p))
            {
                cached.push(p);
            } else {
                discarded.push(p);
            }
        }
        Ok((released, cached, discarded))
    }
    fn retain_scanned(
        &mut self,
        mut arena: PyRefMut<'_, super::ArenaCore>,
        group: usize,
        src: &Bound<'_, PyList>,
        hashes: &mut PrefixHashes,
        page_size: usize,
        count: usize,
        canary: Vec<i64>,
    ) -> PyResult<Vec<usize>> {
        let pages = self.scan(src, hashes, page_size, count, canary)?;
        if arena.retain_many(group, pages.clone())?.is_none() {
            return Err(PyRuntimeError::new_err(
                "cannot retain occupied prefix arena slots",
            ));
        }
        self.retain(pages.clone())?;
        Ok(pages)
    }
    fn release_arena(
        &mut self,
        mut arena: PyRefMut<'_, super::ArenaCore>,
        group: usize,
        pages: Vec<usize>,
    ) -> PyResult<()> {
        arena.check_group(group)?;
        // Validate arena ownership before dropping any request references.
        for &p in &pages {
            self.page(p)?;
            arena.groups[group].extent(p)?;
            if !arena.groups[group].live[p] {
                return Err(PyRuntimeError::new_err("prefix page is not owned by arena"));
            }
        }
        let (released, cached, discarded) = self.release(pages)?;
        if released.is_empty() {
            arena.changed.clear();
            return Ok(());
        }
        arena.discard_cached(group, discarded)?;
        arena.mark_cached(group, cached)?;
        arena.free(group, released)
    }
    fn evict(&mut self, p: usize) -> PyResult<()> {
        let page = self.page(p)?;
        if page.refs != 0 || page.pins != 0 || page.snapshot.is_some() {
            return Err(PyRuntimeError::new_err(
                "cannot evict referenced page or attached snapshot",
            ));
        }
        if let Some(h) = page.hash {
            if self.index.get(&h) == Some(&p) {
                self.index.remove(&h);
            }
        }
        self.untouch(p);
        self.pages[p] = Page::default();
        Ok(())
    }
    fn assert_evictable(&self, p: usize) -> PyResult<()> {
        let page = self.page(p)?;
        if page.refs != 0 || page.pins != 0 {
            return Err(PyRuntimeError::new_err("cannot evict pinned prefix page"));
        }
        Ok(())
    }
    fn __len__(&self) -> usize {
        self.pages.len()
    }
    fn __getitem__(&self, p: usize) -> PyResult<Option<usize>> {
        self.snapshot(p)
    }
    fn mirror_snapshots(&mut self, pages: Vec<usize>, slots: Vec<isize>) -> PyResult<()> {
        if pages.len() != slots.len() {
            return Err(PyRuntimeError::new_err(
                "snapshot assignment length mismatch",
            ));
        }
        for (&p, &slot) in pages.iter().zip(&slots) {
            let page = self.page(p)?;
            let new = if slot < 0 { None } else { Some(slot as usize) };
            if page.pins != 0 && page.snapshot != new {
                return Err(PyRuntimeError::new_err(
                    "cannot replace pinned SSM snapshot",
                ));
            }
        }
        for (p, slot) in pages.into_iter().zip(slots) {
            self.set_snapshot(p, if slot < 0 { None } else { Some(slot as usize) })?;
        }
        Ok(())
    }
    fn snapshot(&self, p: usize) -> PyResult<Option<usize>> {
        Ok(self.page(p)?.snapshot)
    }
    fn set_snapshot(&mut self, p: usize, slot: Option<usize>) -> PyResult<()> {
        let old = self.page(p)?;
        if old.pins != 0 && old.snapshot != slot {
            return Err(PyRuntimeError::new_err(
                "cannot replace pinned SSM snapshot",
            ));
        }
        self.untouch(p);
        self.pages[p].snapshot = slot;
        self.pages[p].valid = false;
        Ok(())
    }
    fn prepare_snapshot(&mut self, p: usize) -> PyResult<Option<usize>> {
        self.page(p)?;
        self.untouch(p);
        self.pages[p].valid = false;
        Ok(self.pages[p].snapshot)
    }
    fn publish(&mut self, p: usize, slot: usize) -> PyResult<()> {
        if self.page(p)?.snapshot == Some(slot) {
            self.pages[p].valid = true;
            self.touch(p);
        }
        Ok(())
    }
    fn valid_snapshot(&mut self, p: usize) -> PyResult<Option<usize>> {
        if !self.page(p)?.valid {
            return Ok(None);
        }
        self.touch(p);
        Ok(self.pages[p].snapshot)
    }
    fn pin(&mut self, p: usize) -> PyResult<()> {
        self.page(p)?;
        self.untouch(p);
        self.pages[p].pins += 1;
        Ok(())
    }
    fn unpin(&mut self, p: usize) -> PyResult<()> {
        if self.page(p)?.pins == 0 {
            return Err(PyRuntimeError::new_err("SSM snapshot pin underflow"));
        }
        self.pages[p].pins -= 1;
        if self.pages[p].pins == 0 {
            self.touch(p);
        }
        Ok(())
    }
    fn oldest_snapshot(&self) -> Option<usize> {
        self.oldest
    }
    fn reclaimable(&self) -> usize {
        self.reclaimable_count
    }
    fn can_release_snapshot(&self, p: usize) -> PyResult<()> {
        if self.page(p)?.pins != 0 {
            return Err(PyRuntimeError::new_err(
                "cannot release pinned SSM snapshot",
            ));
        }
        Ok(())
    }
    fn cacheable(&self, p: usize) -> PyResult<bool> {
        Ok(self.page(p)?.hash.is_some())
    }
    // Diagnostic snapshots are never used in the scheduling hot path.
    fn refs(&self) -> Vec<usize> {
        self.pages.iter().map(|p| p.refs).collect()
    }
    fn hashes(&self) -> Vec<isize> {
        self.pages.iter().map(|p| p.hash.unwrap_or(0)).collect()
    }
    fn canaries(&self) -> Vec<Option<Vec<i64>>> {
        self.pages.iter().map(|p| p.canary.clone()).collect()
    }
    fn set_canary(&mut self, p: usize, canary: Vec<i64>) -> PyResult<()> {
        self.page(p)?;
        self.pages[p].canary = Some(canary);
        Ok(())
    }
    fn snapshots(&self) -> Vec<Option<usize>> {
        self.pages.iter().map(|p| p.snapshot).collect()
    }
    fn validity(&self) -> Vec<bool> {
        self.pages.iter().map(|p| p.valid).collect()
    }
    fn pins(&self) -> HashMap<usize, usize> {
        self.pages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.pins != 0)
            .map(|(i, p)| (i, p.pins))
            .collect()
    }
    fn lru(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.reclaimable_count);
        let mut next = self.oldest;
        while let Some(p) = next {
            result.push(p);
            next = self.pages[p].newer;
        }
        result
    }
    fn index(&self) -> HashMap<isize, usize> {
        self.index.clone()
    }
}

/// Per-request chain. Holding the source prevents Python id reuse; page size
/// changes reset the chain even when the source object itself is unchanged.
#[pyclass(module = "gllm._cache_arena")]
pub struct PrefixHashes {
    source: Py<PyList>,
    page_size: usize,
    hashes: Vec<isize>,
}
#[pymethods]
impl PrefixHashes {
    #[new]
    fn new(source: &Bound<'_, PyList>) -> Self {
        Self {
            source: source.clone().unbind(),
            page_size: 0,
            hashes: Vec::new(),
        }
    }
    fn matches_source(&self, src: &Bound<'_, PyList>) -> bool {
        self.source.bind(src.py()).is(src)
    }
    fn __deepcopy__(&self, py: Python<'_>, memo: &Bound<'_, PyDict>) -> PyResult<Self> {
        let source = py
            .import("copy")?
            .getattr("deepcopy")?
            .call1((self.source.bind(py), memo))?
            .cast_into::<PyList>()?;
        Ok(Self {
            source: source.unbind(),
            page_size: self.page_size,
            hashes: self.hashes.clone(),
        })
    }
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyType>, (Py<PyList>,), (usize, Vec<isize>)) {
        (
            py.get_type::<Self>(),
            (self.source.clone_ref(py),),
            (self.page_size, self.hashes.clone()),
        )
    }
    fn __setstate__(&mut self, py: Python<'_>, state: (usize, Vec<isize>)) -> PyResult<()> {
        let (page_size, hashes) = state;
        if (!hashes.is_empty() && page_size == 0)
            || (page_size != 0 && hashes.len() > self.source.bind(py).len() / page_size)
        {
            return Err(PyRuntimeError::new_err("invalid prefix hash state"));
        }
        self.page_size = page_size;
        self.hashes = hashes;
        Ok(())
    }
    fn __len__(&self) -> usize {
        self.hashes.len()
    }
    fn page_hash(
        &mut self,
        src: &Bound<'_, PyList>,
        page_size: usize,
        idx: usize,
    ) -> PyResult<isize> {
        if page_size == 0 || idx >= src.len() / page_size {
            return Err(PyIndexError::new_err("hash requires a complete page"));
        }
        if !self.source.bind(src.py()).is(src) || self.page_size != page_size {
            self.source = src.clone().unbind();
            self.page_size = page_size;
            self.hashes.clear();
        }
        while self.hashes.len() <= idx {
            let i = self.hashes.len();
            let prev = if i == 0 {
                0x9e3779b97f4a7c15u64.into_pyobject(src.py())?.into_any()
            } else {
                self.hashes[i - 1].into_pyobject(src.py())?.into_any()
            };
            // CPython performs the contiguous slice/copy in C, avoiding one
            // PyO3 bound-object conversion and temporary Vec entry per token.
            let tokens = src.get_slice(i * page_size, (i + 1) * page_size).to_tuple();
            let key = PyTuple::new(src.py(), [prev, tokens.into_any()])?;
            self.hashes.push(key.hash()?);
        }
        Ok(self.hashes[idx])
    }
}
