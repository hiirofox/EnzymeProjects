#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXTERNAL="$ROOT/../../external"
BUILD="$ROOT/builds"

SOURCES=(
    "$ROOT/test.cpp"
    "$EXTERNAL/optim/src/unconstrained/gd.cpp"
)

INCLUDES=(
"$EXTERNAL/eigen"
"$EXTERNAL/optim/include"
)

mkdir -p "$BUILD"

CMD=(
clang++-18
"${SOURCES[@]}"
-std=c++17
-O3 
-DOPTIM_ENABLE_EIGEN_WRAPPERS
-fplugin="$EXTERNAL/ClangEnzyme-18.so"
)

for dir in "${INCLUDES[@]}"; do
CMD+=("-I$dir")
done

CMD+=("-o" "$BUILD/main")

"${CMD[@]}"

echo "Build success: $BUILD/main"
