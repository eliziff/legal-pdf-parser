/// `operation`, timed where LEGALPDF_PHASE_TIMING is set: one line on stderr per measured phase,
/// "phase	<name>	<elapsed_us>", as the citation finder writes its stages.
#[inline(always)]
pub fn measure<T>(name: &'static str, operation: impl FnOnce() -> T) -> T {
    static TIMING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*TIMING.get_or_init(|| std::env::var_os("LEGALPDF_PHASE_TIMING").is_some()) {
        return operation();
    }
    let started = std::time::Instant::now();
    let result = operation();
    eprintln!("phase\t{name}\t{}", started.elapsed().as_micros());
    result
}
