//! Shared-memory arena for media tensors.
//!
//! The front end writes each request's processor outputs into one `/dev/shm`
//! file that every local worker maps; messages carry only offsets. A range is
//! released when its request finishes, so the worker never needs to answer.
//!
//! The file is unlinked as soon as it is created, so nothing is left behind
//! after a crash; workers (including ones restarted after sleep level 2) open
//! it through this process's `/proc/<pid>/fd/<fd>`.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd as _;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context as _, Result};
use memmap2::MmapMut;
use tokio::sync::Notify;

const ALIGN: usize = 256;

/// First-fit allocator over free `[offset, offset + len)` ranges.
#[derive(Debug)]
struct Ranges {
    free: BTreeMap<usize, usize>,
}

impl Ranges {
    fn new(size: usize) -> Self {
        Self { free: BTreeMap::from([(0, size)]) }
    }

    fn alloc(&mut self, len: usize) -> Option<usize> {
        let len = len.max(1).div_ceil(ALIGN) * ALIGN;
        let (&start, &size) = self.free.iter().find(|(_, s)| **s >= len)?;
        self.free.remove(&start);
        if size > len {
            self.free.insert(start + len, size - len);
        }
        Some(start)
    }

    fn release(&mut self, start: usize, len: usize) {
        let mut start = start;
        let mut len = len.max(1).div_ceil(ALIGN) * ALIGN;
        if let Some((&next, &next_len)) = self.free.range(start..).next()
            && next == start + len
        {
            self.free.remove(&next);
            len += next_len;
        }
        if let Some((&prev, &prev_len)) = self.free.range(..start).next_back()
            && prev + prev_len == start
        {
            self.free.remove(&prev);
            start = prev;
            len += prev_len;
        }
        self.free.insert(start, len);
    }
}

pub struct MediaArena {
    /// Where workers open the arena (`engine_server --media-arena`).
    pub path: PathBuf,
    _file: File,
    map: Mutex<MmapMut>,
    ranges: Mutex<Ranges>,
    freed: Notify,
    size: usize,
}

/// One allocated range; returned to the arena with [`MediaArena::release`].
#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub offset: usize,
    pub len: usize,
}

impl MediaArena {
    pub fn create(name: &str, size: usize) -> Result<Self> {
        let path = PathBuf::from(format!("/dev/shm/{name}"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("creating {}", path.display()))?;
        let _ = std::fs::remove_file(&path);
        file.set_len(size as u64)?;
        // SAFETY: the file is ours; workers only read ranges we hand them.
        let map = unsafe { MmapMut::map_mut(&file)? };
        let path = PathBuf::from(format!("/proc/{}/fd/{}", std::process::id(), file.as_raw_fd()));
        Ok(Self {
            path,
            _file: file,
            map: Mutex::new(map),
            ranges: Mutex::new(Ranges::new(size)),
            freed: Notify::new(),
            size,
        })
    }

    /// Copy `data` into a fresh range, waiting up to `wait` for space.
    pub async fn write(&self, data: &[u8], wait: Duration) -> Option<Range> {
        if data.len() > self.size {
            return None;
        }
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let notified = self.freed.notified();
            if let Some(offset) = self.ranges.lock().unwrap().alloc(data.len()) {
                self.map.lock().unwrap()[offset..offset + data.len()].copy_from_slice(data);
                return Some(Range { offset, len: data.len() });
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return None;
            }
        }
    }

    pub fn release(&self, range: Range) {
        self.ranges.lock().unwrap().release(range.offset, range.len);
        self.freed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn arena_is_unlinked_but_reopenable() {
        let name = format!("gllm_test_{}_media", std::process::id());
        let arena = MediaArena::create(&name, 4096).unwrap();
        assert!(!std::path::Path::new(&format!("/dev/shm/{name}")).exists());
        let range = arena.write(b"hello", Duration::from_secs(1)).await.unwrap();
        let reopened = std::fs::read(&arena.path).unwrap();
        assert_eq!(&reopened[range.offset..range.offset + 5], b"hello");
    }

    #[test]
    fn ranges_coalesce_back_into_one_block() {
        let mut r = Ranges::new(4096);
        let a = r.alloc(100).unwrap();
        let b = r.alloc(300).unwrap();
        let c = r.alloc(256).unwrap();
        assert_eq!((a, b, c), (0, 256, 768));
        r.release(b, 300);
        assert_eq!(r.alloc(512), Some(256)); // reuses the hole
        r.release(256, 512);
        r.release(a, 100);
        r.release(c, 256);
        assert_eq!(r.free.len(), 1);
        assert_eq!(r.free[&0], 4096);
        assert_eq!(r.alloc(4097), None);
    }
}
