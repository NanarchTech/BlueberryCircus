#!/bin/sh
# Build the BlueberryCircus Rust LL-RK4 integrator (dependency-free cdylib).
# The Python RustBackend loads target/release/libblueberry_rs.{dylib,so} via ctypes.
set -e
cd "$(CDPATH= cd "$(dirname "$0")/../rust/blueberry_rs" && pwd)"
cargo build --release
echo "built in: $(pwd)/target/release/"
ls -1 target/release/libblueberry_rs.* 2>/dev/null || true
