use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let provider = args.next().and_then(|value| value.into_string().ok());
    let source = args.next().map(PathBuf::from);
    let target = args.next().map(PathBuf::from);
    let limit = args
        .next()
        .and_then(|value| value.to_string_lossy().parse().ok());
    if provider.as_deref() != Some("a2aj") || source.is_none() || target.is_none() {
        return Err("usage: legal-structure-store a2aj SOURCE.sqlite TARGET.sqlite [LIMIT]".into());
    }
    let summary =
        legal_structure_store::a2aj::import(source.unwrap(), target.unwrap(), limit, |count| {
            eprintln!("materialized {count} SourceDocs")
        })?;
    eprintln!(
        "materialized {} SourceDocs total ({} this run, {})",
        summary.total,
        summary.processed,
        if summary.complete {
            "complete"
        } else {
            "partial"
        }
    );
    Ok(())
}
