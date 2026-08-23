use legal_structure::{derive_definitions, DefinitionParagraph};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
struct Input {
    id: String,
    text: String,
    paragraphs: Vec<DefinitionParagraph>,
}

#[derive(Serialize)]
struct Output<T> {
    id: String,
    result: Option<T>,
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let input = serde_json::from_str::<Input>(&line)?;
        let output = Output {
            id: input.id,
            result: Some(derive_definitions(&input.text, &input.paragraphs)),
            error: None,
        };
        serde_json::to_writer(&mut stdout, &output)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
