#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXTERNAL="$ROOT/../../external"
BUILD="$ROOT/builds"

WINDOWS_TARGET=false

SOURCES=(
    "$ROOT/test.cpp"
    "$EXTERNAL/optim/src/unconstrained/gd.cpp"
    "$EXTERNAL/optim/src/unconstrained/lbfgs.cpp"
    "$EXTERNAL/optim/src/line_search/more_thuente.cpp"
)

INCLUDES=(
    "$EXTERNAL/eigen"
    "$EXTERNAL/optim/include"
    "$EXTERNAL/ensmallen/include"
)

mkdir -p "$BUILD"

CMD=(
    clang++-18
    "${SOURCES[@]}"

    -larmadillo

    -std=c++17
    -O3 -march=native -fopenmp
    #-ffast-math
    -DOPTIM_ENABLE_EIGEN_WRAPPERS
    -fplugin="$EXTERNAL/ClangEnzyme-18.so"
)

if [ "$WINDOWS_TARGET" = true ]; then
    CMD+=(
        --target=x86_64-w64-windows-gnu
        -static
    )
    OUTPUT="$BUILD/main.exe"
else
    OUTPUT="$BUILD/main"
fi

for dir in "${INCLUDES[@]}"; do
    CMD+=("-I$dir")
done

CMD+=("-o" "$OUTPUT")

"${CMD[@]}"

echo "Build success: $OUTPUT"