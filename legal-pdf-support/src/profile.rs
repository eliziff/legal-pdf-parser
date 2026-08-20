#[cfg(feature = "profiling")]
mod enabled {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    struct CountingAllocator;

    static COUNTING: AtomicBool = AtomicBool::new(false);
    static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
    static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() && COUNTING.load(Ordering::Relaxed) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            if COUNTING.load(Ordering::Relaxed) {
                DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            unsafe { System.dealloc(pointer, layout) };
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let result = unsafe { System.realloc(pointer, layout, new_size) };
            if !result.is_null() && COUNTING.load(Ordering::Relaxed) {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
                DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            result
        }
    }

    fn snapshot() -> (u64, u64, u64) {
        (
            ALLOCATIONS.load(Ordering::Relaxed),
            ALLOCATED_BYTES.load(Ordering::Relaxed),
            DEALLOCATED_BYTES.load(Ordering::Relaxed),
        )
    }

    pub(super) fn begin() {
        COUNTING.store(
            std::env::var_os("LEGALPDF_PROFILE_ALLOC").is_some(),
            Ordering::Relaxed,
        );
    }

    pub(super) fn end() {
        COUNTING.store(false, Ordering::Relaxed);
    }

    pub(super) fn measure<T>(name: &str, operation: impl FnOnce() -> T) -> T {
        if std::env::var_os("LEGALPDF_PROFILE_PHASES").is_none() {
            return operation();
        }
        let before = snapshot();
        let started = Instant::now();
        let result = operation();
        let elapsed = started.elapsed().as_secs_f64();
        let after = snapshot();
        eprintln!(
            "LEGALPDF_PHASE name={name} seconds={elapsed:.6} allocations={} allocated_bytes={} deallocated_bytes={}",
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
        );
        result
    }
}

#[inline(always)]
pub fn begin() {
    #[cfg(feature = "profiling")]
    enabled::begin();
}

#[inline(always)]
pub fn end() {
    #[cfg(feature = "profiling")]
    enabled::end();
}

#[inline(always)]
pub fn measure<T>(name: &str, operation: impl FnOnce() -> T) -> T {
    #[cfg(feature = "profiling")]
    {
        enabled::measure(name, operation)
    }
    #[cfg(not(feature = "profiling"))]
    {
        let _ = name;
        operation()
    }
}
