fn main() {
    if let Err(error) = legal_structure::native_stdio_sidecar() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
