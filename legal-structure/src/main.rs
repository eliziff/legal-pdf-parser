fn main() {
    if let Err(error) = legal_structure::stdio_sidecar() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
