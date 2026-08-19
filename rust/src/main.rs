#[cfg(feature = "fast-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use legalpdf::{document_request, extract_common_input, replay_common_input, Error, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const MAX_REQUEST_BYTES: u64 = 64 * 1024;

fn usage() -> &'static str {
    "usage:\n  legalpdf contract <request.json>\n  legalpdf --version"
}

fn take_value(arguments: &[String], index: &mut usize, option: &str) -> Result<String> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| Error::Message(format!("{option} requires a value")))
}

fn hidden_output(arguments: &[String], command: &str) -> Result<(PathBuf, PathBuf)> {
    let input = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(format!("{command} requires an input")))?;
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?))
            }
            option => {
                return Err(Error::Message(format!(
                    "unknown {command} option: {option}"
                )))
            }
        }
        index += 1;
    }
    Ok((
        input,
        output.ok_or_else(|| Error::Message(format!("{command} requires --output <path>")))?,
    ))
}

fn parity_extract_command(arguments: &[String]) -> Result<i32> {
    let (input, output) = hidden_output(arguments, "_parity-extract")?;
    let path = extract_common_input(input, output)?;
    println!("{}", serde_json::to_string(&json!({"result": path}))?);
    Ok(0)
}

fn parity_replay_command(arguments: &[String]) -> Result<i32> {
    let (input, output) = hidden_output(arguments, "_parity-replay")?;
    let path = replay_common_input(input, output)?;
    println!("{}", serde_json::to_string(&json!({"result": path}))?);
    Ok(0)
}

fn resolve_request_path(value: &mut Value, key: &str, base: &Path) {
    let Some(path) = value.get(key).and_then(Value::as_str).map(PathBuf::from) else {
        return;
    };
    if path.is_relative() {
        value[key] = Value::String(base.join(path).to_string_lossy().into_owned());
    }
}

fn resolve_request_paths(value: &mut Value, input: &Path) {
    let base = input.parent().unwrap_or_else(|| Path::new("."));
    resolve_request_path(value, "source_pdf", base);
    resolve_request_path(value, "cache_dir", base);
    for (provider, fields) in [
        (
            "ocr",
            &[
                "command",
                "model",
                "codec",
                "runtime",
                "runtime_wheel",
                "python",
                "blla_pack",
                "recognizer_pack",
                "tesseract_library",
            ][..],
        ),
        ("layout", &["model_pack", "runtime", "cache_dir"][..]),
    ] {
        if let Some(settings) = value
            .get_mut(provider)
            .and_then(|provider| provider.get_mut("settings"))
        {
            for field in fields {
                resolve_request_path(settings, field, base);
            }
        }
    }
}

fn progress(operation: &str, phase: &str, completed: usize, total: usize) -> Result<()> {
    eprintln!(
        "{}",
        serde_json::to_string(&json!({
            "schema_version": "legalpdf.progress.v1",
            "operation": operation,
            "phase": phase,
            "completed": completed,
            "total": total,
        }))?
    );
    Ok(())
}

fn contract_command(arguments: &[String]) -> Result<i32> {
    let input = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message("contract requires an input".to_owned()))?;
    if arguments.len() != 1 {
        return Err(Error::Message(
            "contract accepts exactly one input".to_owned(),
        ));
    }
    let metadata = std::fs::metadata(&input).map_err(|source| Error::io(&input, source))?;
    if metadata.len() > MAX_REQUEST_BYTES {
        return Err(Error::Message("contract input exceeds 64 KiB".to_owned()));
    }
    let bytes = std::fs::read(&input).map_err(|source| Error::io(&input, source))?;
    let mut value: Value = serde_json::from_slice(&bytes)?;
    resolve_request_paths(&mut value, &input);
    let operation = value
        .get("operation")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let total = if operation == "prepare" {
        value
            .get("pages")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0)
    } else {
        0
    };
    if operation == "prepare" {
        progress(operation, "preparing", 0, total)?;
    }
    let result = document_request(&value)?;
    if operation == "prepare" {
        let completed = result
            .pointer("/source/page_count")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(total);
        progress(operation, "ready", completed, completed)?;
    }
    println!("{}", serde_json::to_string(&result)?);
    Ok(0)
}

fn run() -> Result<i32> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = arguments
        .split_first()
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    match command.as_str() {
        "contract" => contract_command(rest),
        "_parity-extract" => parity_extract_command(rest),
        "_parity-replay" => parity_replay_command(rest),
        "--version" | "-V" => {
            println!("legalpdf {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        "--help" | "-h" => {
            println!("{}", usage());
            Ok(0)
        }
        _ => Err(Error::Message(usage().to_owned())),
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("legalpdf: {error}");
            std::process::exit(1);
        }
    }
}
