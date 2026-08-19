use crate::artifact::python_json;
use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub(crate) fn stable_hash(value: &Value) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn command_path() -> Option<PathBuf> {
    if let Ok(command) = env::var("CODEX_EXEC_COMMAND") {
        let command = command.trim();
        if !command.is_empty() {
            return Some(PathBuf::from(command));
        }
    }
    let path = env::var_os("PATH")?;
    #[cfg(windows)]
    let extensions = env::var_os("PATHEXT")
        .map(|value| {
            value
                .to_string_lossy()
                .split(';')
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![".COM".into(), ".EXE".into(), ".BAT".into(), ".CMD".into()]);
    for directory in env::split_paths(&path) {
        let plain = directory.join("codex");
        if plain.is_file() {
            return Some(plain);
        }
        #[cfg(windows)]
        for extension in &extensions {
            let candidate = directory.join(format!("codex{extension}"));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn usage_from_events(stdout: &str) -> BTreeMap<String, i64> {
    fn visit(value: &Value, totals: &mut BTreeMap<String, i64>) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    if matches!(
                        key.as_str(),
                        "input_tokens" | "output_tokens" | "cached_input_tokens" | "total_tokens"
                    ) {
                        if let Some(value) = child.as_i64() {
                            totals
                                .entry(key.clone())
                                .and_modify(|current| *current = (*current).max(value))
                                .or_insert(value);
                        }
                    } else {
                        visit(child, totals);
                    }
                }
            }
            Value::Array(values) => {
                for value in values {
                    visit(value, totals);
                }
            }
            _ => {}
        }
    }
    let mut totals = BTreeMap::new();
    for line in stdout.lines() {
        if let Ok(value) = serde_json::from_str(line) {
            visit(&value, &mut totals);
        }
    }
    totals
}

pub(crate) fn invoke(
    prompt: &str,
    schema_path: &Path,
    image_paths: &[PathBuf],
    model: &str,
    effort: &str,
    work_dir: &Path,
    timeout_seconds: u64,
) -> Result<(Value, BTreeMap<String, i64>, f64)> {
    let executable = command_path()
        .ok_or_else(|| Error::Message("codex executable was not found on PATH".to_owned()))?;
    let output_path = work_dir.join("last-message.json");
    match fs::remove_file(&output_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => return Err(Error::io(&output_path, source)),
    }
    let effort_json = python_json(&Value::String(effort.to_owned()))?;
    let mut command = Command::new(&executable);
    command
        .args([
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "-c",
        ])
        .arg(format!("model_reasoning_effort={effort_json}"))
        .arg("--output-schema")
        .arg(schema_path)
        .arg("--output-last-message")
        .arg(&output_path)
        .args(["--color", "never", "--json"]);
    for image in image_paths {
        command.arg("--image").arg(image);
    }
    command
        .arg("-")
        .current_dir(work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|source| Error::io(&executable, source))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::Message("codex stdout was not captured".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| Error::Message("codex stderr was not captured".to_owned()))?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut reader = BufReader::new(stdout);
        reader.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut reader = BufReader::new(stderr);
        reader.read_to_end(&mut bytes).map(|_| bytes)
    });
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .map_err(|source| Error::io(&executable, source))?;
    }
    let timeout = Duration::from_secs(timeout_seconds);
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|source| Error::io(&executable, source))?
        {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(Error::Message(format!(
                "codex exec timed out after {timeout_seconds} seconds"
            )));
        }
        thread::sleep(Duration::from_millis(20));
    };
    let elapsed = started.elapsed().as_secs_f64();
    let stdout = stdout_reader
        .join()
        .map_err(|_| Error::Message("codex stdout reader failed".to_owned()))?
        .map_err(|source| Error::io(&executable, source))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| Error::Message("codex stderr reader failed".to_owned()))?
        .map_err(|source| Error::io(&executable, source))?;
    let stdout = String::from_utf8_lossy(&stdout).into_owned();
    let stderr = String::from_utf8_lossy(&stderr).into_owned();
    if !status.success() {
        let message = if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };
        let tail = message
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>();
        return Err(Error::Message(format!(
            "codex exec exited with {}: {tail}",
            status.code().unwrap_or(-1)
        )));
    }
    if !output_path.is_file() {
        return Err(Error::Message(
            "codex exec did not write its final response".to_owned(),
        ));
    }
    let bytes = fs::read(&output_path).map_err(|source| Error::io(&output_path, source))?;
    let response = serde_json::from_slice(&bytes)
        .map_err(|error| Error::Message(format!("codex response is not valid JSON: {error}")))?;
    Ok((response, usage_from_events(&stdout), elapsed))
}
