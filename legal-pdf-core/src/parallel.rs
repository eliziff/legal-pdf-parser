//! Work over a document's parts on every core.

/// `work` over `items` on every core where there are threads (natively), in order where there are none
/// (WebAssembly): each result in its item's place. The cores' threads are one pool that lives as long as the program,
/// so what a thread keeps for its work (a pattern's scratch space, a reader's tables) is made once per thread, not once
/// per call. A panic in `work` is the caller's.
pub fn each<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        if items.len() > 1 {
            return items
                .par_iter()
                .map(&work)
                .collect();
        }
    }
    items.iter().map(work).collect()
}
