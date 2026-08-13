# Contributing

Issues and reviewed pull requests are welcome. There is no response-time
guarantee. This is a research prototype maintained alongside other work.

## Ground rules

- **Certificates are the contract.** Every reported quantitative result carries
  a `PASS`/`FAIL`/`NULL` certificate re-derivable from its recorded numbers. A
  change that weakens a rule, silences a check, or converts a `FAIL` into a
  skip will not be merged. A skipped mandatory check counts as a failure.
- **Honest verdicts.** `xfail(strict=True)` markers document results the
  literature says cannot pass (stable hydrogen) or that need CPU-day compute
  (O3 convergence). Do not "fix" them by loosening tolerances.
- **numpy is the trust root.** The Rust and JAX backends must agree with the
  numpy reference under the existing enclosure tests; new backends need the
  same cross-language gate.
- **One runtime dependency.** Keep the core importable with numpy alone.

## Workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # core suite: must be 89 passed, 3 xfailed, 0 skipped
sh scripts/build_rust.sh && pytest -m rust   # if your change touches the integrator
pip install ".[jax]" && pytest -m jax        # if your change touches backends
```

PRs should state which oracles/certificates cover the change and include test
output. Physics changes need a literature citation or an analytic derivation in
`docs/theory.md`.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
