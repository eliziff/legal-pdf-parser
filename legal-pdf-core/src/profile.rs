use std::cell::Cell;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn timing() -> bool {
    static TIMING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TIMING.get_or_init(|| std::env::var_os("LEGALPDF_PHASE_TIMING").is_some())
}

/// A step's time: its whole, and its own (the whole less the timed steps run inside it on its thread).
#[derive(Clone, Copy, Default)]
struct Spent {
    total: Duration,
    own: Duration,
}

type Tallies = Arc<Mutex<Vec<(&'static str, Spent)>>>;

/// Every thread's tallies (`MINE`), for the writer (`tallied`) to read them all.
static THREADS: Mutex<Vec<Tallies>> = Mutex::new(Vec::new());

thread_local! {
    /// Each step's time so far on this thread, summed over its calls, until written (`tallied`): its own lock, which no
    /// other thread takes but the writer's.
    static MINE: Tallies = {
        let mine = Tallies::default();
        THREADS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(Arc::clone(&mine));
        mine
    };
    /// The time of the timed steps run inside the step now running on this thread.
    static INSIDE: Cell<Duration> = const { Cell::new(Duration::ZERO) };
}

/// `operation` timed: its whole time, and its own, the time of the timed steps inside it taken off.
fn timed<T>(operation: impl FnOnce() -> T) -> (T, Spent) {
    let outer = INSIDE.with(|inside| inside.replace(Duration::ZERO));
    let started = Instant::now();
    let result = operation();
    let total = started.elapsed();
    let inner = INSIDE.with(|inside| inside.replace(outer + total));
    (result, Spent { total, own: total.saturating_sub(inner) })
}

/// `operation`, timed where LEGALPDF_PHASE_TIMING is set: its time summed under `name` with its every other call's and
/// written once (`tallied`), "phase	<name>	<elapsed_us>	<own_us>". A line written per call costs a step run often
/// more than the step: the timing would time its own writing.
#[inline(always)]
pub fn measure<T>(name: &'static str, operation: impl FnOnce() -> T) -> T {
    tally(name, operation)
}

/// A parse's document, its size written where LEGALPDF_PHASE_TIMING is set ("document	<pages>	<lines>"): what a
/// stage's time is read against.
pub fn document(pages: usize, lines: usize) {
    if timing() {
        eprintln!("document\t{pages}\t{lines}");
    }
}

/// `operation`, its time added to `name`'s where LEGALPDF_PHASE_TIMING is set: for a step run on every line, whose times
/// are written once, summed (`tallied`), not a line each.
#[inline(always)]
pub fn tally<T>(name: &'static str, operation: impl FnOnce() -> T) -> T {
    if !timing() {
        return operation();
    }
    let (result, spent) = timed(operation);
    MINE.with(|mine| add(&mut mine.lock().unwrap_or_else(|poisoned| poisoned.into_inner()), name, spent));
    result
}

fn add(tallies: &mut Vec<(&'static str, Spent)>, name: &'static str, spent: Spent) {
    match tallies.iter_mut().find(|(n, _)| *n == name) {
        Some((_, sum)) => {
            sum.total += spent.total;
            sum.own += spent.own;
        }
        None => tallies.push((name, spent)),
    }
}

/// The tallied steps' times so far, every thread's, written as phases ("phase	<name>	<elapsed_us>	<own_us>") and
/// cleared.
pub fn tallied() {
    if !timing() {
        return;
    }
    let mut drained = Vec::new();
    for tallies in THREADS.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).iter() {
        for (name, spent) in std::mem::take(&mut *tallies.lock().unwrap_or_else(|poisoned| poisoned.into_inner())) {
            add(&mut drained, name, spent);
        }
    }
    for (name, spent) in drained {
        eprintln!("phase\t{name}\t{}\t{}", spent.total.as_micros(), spent.own.as_micros());
    }
}
