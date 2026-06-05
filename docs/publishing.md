# Publishing

The Rust crate is published to [crates.io](https://crates.io/crates/pdf-inspector) with trusted publishing from GitHub Actions. The first release was published manually; future releases publish from `.github/workflows/publish-crate.yml` when a `Cargo.toml` version change lands on `main`.

## crates.io Trusted Publisher

Configure the trusted publisher for the `pdf-inspector` crate with:

- Repository: `firecrawl/pdf-inspector`
- Workflow: `publish-crate.yml`
- Environment: `crates-io`

The workflow uses `rust-lang/crates-io-auth-action@v1` to exchange GitHub's OIDC token for a short-lived crates.io token, then passes it to `cargo publish`.

## Release Steps

1. Update `version` in `Cargo.toml`.
2. Merge the version bump to `main`.
3. The publish workflow compares the new `Cargo.toml` version with `HEAD~1`, runs `cargo publish --dry-run`, then publishes if that version is not already on crates.io.

If `Cargo.toml` changes without a package version bump, the workflow exits without publishing.
