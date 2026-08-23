# N-API result transport

Release addon, warmed calls, median wall time. Every path ran the same Rust
`instrument` request; Buffer timing includes UTF-8 decoding and `JSON.parse`.

| Input/output bytes | Direct object | JSON Buffer | Buffer change |
| ---: | ---: | ---: | ---: |
| 2,925 / 3,840 | 1.431 ms | 1.397 ms | -2.4% |
| 35,525 / 107,301 | 8.114 ms | 7.931 ms | -2.3% |
| 1,013,485 / 1,629,996 | 367.962 ms | 325.331 ms | -11.6% |

A confirmatory run also favored Buffer by 2.6%, 11.1%, and 7.6%. Request
conversion plus typed-enum decoding was only 0.045, 0.153, and 2.722 ms
(3.2%, 1.9%, and 0.7% of the direct calls). Values were deep-equal.

Direct JSON String was effectively tied under detector noise; on the large
fixture Buffer remained 1.7% faster by paired median. Keep one UTF-8 Buffer
operation and parse it once in JavaScript.

Large fixture: 1,008,540 Unicode scalars/UTF-16 units and 1,013,485 UTF-8
bytes. Options were `reconstruct_lineation: true`, no table cells, and no
SourceDoc projection.

Commands used: `cargo build --release --offline -p legal-structure-node`, then
`node --expose-gc legal-structure-node/experiments/ffi-transport/bench.mjs`.
