#[cfg(feature = "profiling")]
mod enabled {
    #[cfg(feature = "allocation-profiling")]
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::{AtomicU64, Ordering};
    #[cfg(feature = "allocation-profiling")]
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;

    #[cfg(feature = "allocation-profiling")]
    struct CountingAllocator;

    #[cfg(feature = "allocation-profiling")]
    static COUNTING: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "allocation-profiling")]
    static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
    #[cfg(feature = "allocation-profiling")]
    static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
    #[cfg(feature = "allocation-profiling")]
    static DEALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
    static RUNS: AtomicU64 = AtomicU64::new(0);

    thread_local! {
        static RUN: Cell<u64> = const { Cell::new(0) };
        static DEPTH: Cell<u32> = const { Cell::new(0) };
        static RECORDS: RefCell<Vec<Record>> = const { RefCell::new(Vec::new()) };
    }

    struct Record {
        name: &'static str,
        run: u64,
        depth: u32,
        elapsed_us: u128,
        allocations: u64,
        allocated_bytes: u64,
        deallocated_bytes: u64,
        allocation_tracking: bool,
    }

    #[cfg(feature = "allocation-profiling")]
    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    #[cfg(feature = "allocation-profiling")]
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() && COUNTING.load(Ordering::Relaxed) != 0 {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            if COUNTING.load(Ordering::Relaxed) != 0 {
                DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            unsafe { System.dealloc(pointer, layout) };
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let result = unsafe { System.realloc(pointer, layout, new_size) };
            if !result.is_null() && COUNTING.load(Ordering::Relaxed) != 0 {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
                DEALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            result
        }
    }

    #[cfg(feature = "allocation-profiling")]
    fn snapshot() -> (u64, u64, u64) {
        (
            ALLOCATIONS.load(Ordering::Relaxed),
            ALLOCATED_BYTES.load(Ordering::Relaxed),
            DEALLOCATED_BYTES.load(Ordering::Relaxed),
        )
    }

    #[cfg(not(feature = "allocation-profiling"))]
    fn snapshot() -> (u64, u64, u64) {
        (0, 0, 0)
    }

    #[cfg(feature = "allocation-profiling")]
    fn start_allocations() -> bool {
        let enabled = std::env::var_os("LEGALPDF_PROFILE_ALLOC").is_some();
        if enabled {
            COUNTING.fetch_add(1, Ordering::Relaxed);
        }
        enabled
    }

    #[cfg(not(feature = "allocation-profiling"))]
    fn start_allocations() -> bool {
        false
    }

    #[cfg(feature = "allocation-profiling")]
    fn stop_allocations(enabled: bool) {
        if enabled {
            COUNTING.fetch_sub(1, Ordering::Relaxed);
        }
    }

    #[cfg(not(feature = "allocation-profiling"))]
    fn stop_allocations(_: bool) {}

    fn allocations_enabled() -> bool {
        cfg!(feature = "allocation-profiling")
            && std::env::var_os("LEGALPDF_PROFILE_ALLOC").is_some()
    }

    pub(super) struct Scope {
        name: &'static str,
        run: u64,
        started: Instant,
        before: (u64, u64, u64),
        allocations: bool,
    }

    impl Scope {
        pub(super) fn new(name: &'static str) -> Self {
            let run = RUNS.fetch_add(1, Ordering::Relaxed) + 1;
            let allocations = start_allocations();
            RUN.set(run);
            DEPTH.set(0);
            RECORDS.with(|records| {
                let mut records = records.borrow_mut();
                records.clear();
                if records.capacity() < 64 {
                    let additional = 64 - records.capacity();
                    records.reserve(additional);
                }
            });
            Self {
                name,
                run,
                started: Instant::now(),
                before: snapshot(),
                allocations,
            }
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            let after = snapshot();
            let elapsed_us = self.started.elapsed().as_micros();
            let records = RECORDS.with(|records| std::mem::take(&mut *records.borrow_mut()));
            RUN.set(0);
            DEPTH.set(0);
            stop_allocations(self.allocations);
            eprintln!(
                "LEGALPDF_PROFILE event=start run={} name={} allocations={}",
                self.run, self.name, self.allocations,
            );
            for record in records {
                eprintln!(
                    "LEGALPDF_PROFILE event=span run={} depth={} name={} elapsed_us={} inclusive=true allocations={} allocated_bytes={} deallocated_bytes={} allocation_scope=process timing_distorted_by_allocations={}",
                    record.run,
                    record.depth,
                    record.name,
                    record.elapsed_us,
                    record.allocations,
                    record.allocated_bytes,
                    record.deallocated_bytes,
                    record.allocation_tracking,
                );
            }
            eprintln!(
                "LEGALPDF_PROFILE event=end run={} name={} elapsed_us={} allocations={} allocated_bytes={} deallocated_bytes={} allocation_scope=process timing_distorted_by_allocations={}",
                self.run,
                self.name,
                elapsed_us,
                after.0 - self.before.0,
                after.1 - self.before.1,
                after.2 - self.before.2,
                self.allocations,
            );
        }
    }

    struct Span {
        name: &'static str,
        run: u64,
        depth: u32,
        started: Instant,
        before: (u64, u64, u64),
        allocations: bool,
    }

    impl Span {
        fn new(name: &'static str) -> Self {
            let depth = DEPTH.get();
            DEPTH.set(depth + 1);
            Self {
                name,
                run: RUN.get(),
                depth,
                started: Instant::now(),
                before: snapshot(),
                allocations: allocations_enabled(),
            }
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            DEPTH.set(self.depth);
            let after = snapshot();
            RECORDS.with(|records| {
                records.borrow_mut().push(Record {
                    name: self.name,
                    run: self.run,
                    depth: self.depth,
                    elapsed_us: self.started.elapsed().as_micros(),
                    allocations: after.0 - self.before.0,
                    allocated_bytes: after.1 - self.before.1,
                    deallocated_bytes: after.2 - self.before.2,
                    allocation_tracking: self.allocations,
                })
            });
        }
    }

    pub(super) fn measure<T>(name: &'static str, operation: impl FnOnce() -> T) -> T {
        if std::env::var_os("LEGALPDF_PROFILE_PHASES").is_none() {
            return operation();
        }
        let _span = Span::new(name);
        operation()
    }
}

pub struct Scope {
    #[cfg(feature = "profiling")]
    _inner: Option<enabled::Scope>,
}

pub fn scope(name: &'static str) -> Scope {
    #[cfg(feature = "profiling")]
    let inner = std::env::var_os("LEGALPDF_PROFILE_PHASES")
        .is_some()
        .then(|| enabled::Scope::new(name));
    #[cfg(not(feature = "profiling"))]
    let _ = name;
    Scope {
        #[cfg(feature = "profiling")]
        _inner: inner,
    }
}

#[inline(always)]
pub fn measure<T>(name: &'static str, operation: impl FnOnce() -> T) -> T {
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
