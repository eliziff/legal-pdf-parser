use std::io::{self, Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(windows)]
fn configure_process_group(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
}

#[cfg(not(any(unix, windows)))]
fn configure_process_group(_: &mut Command) {}

#[cfg(unix)]
fn terminate_process_tree(child: &mut std::process::Child) {
    // The child is its process-group leader, so this also closes pipes held by
    // descendants before the reader threads are joined.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
}

#[cfg(windows)]
fn terminate_process_tree(child: &mut std::process::Child) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .status();
    let _ = child.kill();
}

#[cfg(not(any(unix, windows)))]
fn terminate_process_tree(child: &mut std::process::Child) {
    let _ = child.kill();
}

pub(crate) struct Output {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub stdout_exceeded: bool,
}

pub(crate) enum RunError {
    Io(io::Error),
    Timeout,
}

fn read_bounded(
    mut reader: impl Read,
    limit: usize,
    keep_tail: bool,
    exceeded_signal: Option<&AtomicBool>,
) -> io::Result<(Vec<u8>, bool)> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut exceeded = false;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let over_limit = output.len().saturating_add(read) > limit;
        exceeded |= over_limit;
        if over_limit {
            if let Some(signal) = exceeded_signal {
                signal.store(true, Ordering::Relaxed);
            }
        }
        if keep_tail {
            output.extend_from_slice(&buffer[..read]);
            if output.len() > limit {
                output.drain(..output.len() - limit);
            }
        } else if output.len() < limit {
            output.extend_from_slice(&buffer[..read.min(limit - output.len())]);
        }
    }
    Ok((output, exceeded))
}

fn join<T>(handle: thread::JoinHandle<io::Result<T>>) -> io::Result<T> {
    handle
        .join()
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "child-process I/O thread failed"))?
}

pub(crate) fn run(
    mut command: Command,
    input: Vec<u8>,
    timeout: Duration,
    stdout_limit: usize,
    stderr_limit: usize,
) -> Result<Output, RunError> {
    configure_process_group(&mut command);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command.spawn().map_err(RunError::Io)?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let stdout_exceeded = Arc::new(AtomicBool::new(false));
    let writer = thread::spawn(move || stdin.write_all(&input));
    let stdout_signal = Arc::clone(&stdout_exceeded);
    let stdout =
        thread::spawn(move || read_bounded(stdout, stdout_limit, false, Some(&stdout_signal)));
    let stderr = thread::spawn(move || read_bounded(stderr, stderr_limit, true, None));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {}
            Err(error) => {
                terminate_process_tree(&mut child);
                let _ = child.wait();
                break Err(RunError::Io(error));
            }
        }
        if stdout_exceeded.load(Ordering::Relaxed) {
            terminate_process_tree(&mut child);
            break child.wait().map_err(RunError::Io);
        }
        if started.elapsed() >= timeout {
            terminate_process_tree(&mut child);
            let _ = child.wait();
            break Err(RunError::Timeout);
        }
        thread::sleep(Duration::from_millis(20));
    };
    let writer = join(writer);
    let stdout = join(stdout);
    let stderr = join(stderr);
    let (stdout, stdout_exceeded) = stdout.map_err(RunError::Io)?;
    let (stderr, _) = stderr.map_err(RunError::Io)?;
    let status = status?;
    if status.success() && !stdout_exceeded {
        writer.map_err(RunError::Io)?;
    }
    Ok(Output {
        status,
        stdout,
        stderr,
        stdout_exceeded,
    })
}
