#[cfg(feature = "fast-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use legalpdf::{digest_cached_extraction, Error, Result};
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage:\n  legalpdf --version"
}

fn parity_replay_batch_command(arguments: &[String]) -> Result<i32> {
    let [manifest] = arguments else {
        return Err(Error::Message(
            "_parity-replay-batch requires <manifest.json>".to_owned(),
        ));
    };
    let path = PathBuf::from(manifest);
    let bytes = std::fs::read(&path).map_err(|source| Error::io(&path, source))?;
    if bytes.len() > 1024 * 1024 {
        return Err(Error::Message("replay manifest exceeds 1 MiB".to_owned()));
    }
    let jobs: Vec<(PathBuf, String)> = serde_json::from_slice(&bytes)?;
    for (index, batch) in jobs.chunks(25).enumerate() {
        for (input, source_name) in batch {
            let value = digest_cached_extraction(input, source_name.clone())?;
            println!("{}", serde_json::to_string(&value)?);
        }
        let completed = ((index + 1) * 25).min(jobs.len());
        eprintln!("replayed {completed}/{}", jobs.len());
    }
    Ok(0)
}

fn run() -> Result<i32> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = arguments
        .split_first()
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    match command.as_str() {
        "_parity-replay-batch" => parity_replay_batch_command(rest),
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
