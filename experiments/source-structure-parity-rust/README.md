# Rust source-structure parity runner

This runner reads the completed TypeScript freeze as an
immutable oracle; it never creates or updates baseline artifacts.

```powershell
cargo run --release -p source-structure-parity-rust -- `
  --baseline ..\backend\experiments\source-structure-parity\results\installed-provider-freeze-full `
  --journal-db "C:\path\to\journals.db" `
  --journal-final-db "C:\path\to\oajd\journals.db"
```

The default time limit is 180 seconds. Use `--provider a2aj` and `--limit 100`
for a bounded check while developing.

A2AJ and CourtListener require exact frozen bytes. Changed journal outputs are
reported separately because both journal branches were deliberately replaced:
`pages.jsonl` uses only `type: "text"` regions as ordinary paragraphs, while
stored plaintext uses the existing unnumbered-prose delineation. Every journal
row must still compile successfully; unchanged rows remain exact matches.
