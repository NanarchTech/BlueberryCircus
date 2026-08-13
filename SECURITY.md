# Security Policy

BlueberryCircus is a numerical simulation library with a single runtime
dependency (numpy). It performs no network I/O; file I/O is limited to
reading/writing certificate JSON bundles at caller-supplied paths, and the
optional Rust backend loads a locally built `cdylib` via `ctypes`.

## Reporting

Report suspected vulnerabilities privately via GitHub's *Report a
vulnerability* (Security Advisories) on this repository rather than a public
issue. Include a minimal reproduction. You should receive an acknowledgement,
but there is no guaranteed response time.

## Scope notes

- Certificate bundles are data, not code; `load_bundle` parses JSON only.
  Tampered bundles are expected inputs and must be *rejected by verdict*, not
  trusted. A bundle that re-derives inconsistently is a `FAIL`.
- Do not point `BLUEBERRY_VERIFY_BIN` or `BLUEBERRY_CERTIFY_SRC` at untrusted
  binaries/directories: both are executed/imported with your privileges.
