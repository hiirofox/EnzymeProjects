# TestEnzyme

一个基于 C++、LLVM/Clang、Enzyme 和 OptimLib 的自动微分与数值优化开发环境。

本项目主要使用 Enzyme 的 Clang 插件，让普通 C/C++ 代码能够直接通过 LLVM 进行自动微分。

ChatGPT指导环境搭建与调试。目前结构已成型。

## 环境

当前环境基于：

- Windows + WSL2
- Ubuntu
- LLVM 18
- Clang 18
- Enzyme
- Eigen 3.4
- OptimLib
- Ninja
- CMake

## 1. 安装 WSL2

在 Windows 管理员 PowerShell 中执行：

```powershell
wsl --install
```

重启 Windows 后进入 Ubuntu。

检查是否运行在 WSL2：

```powershell
wsl -l -v
```

应看到类似：

```text
NAME      STATE     VERSION
Ubuntu    Running   2
```

## 2. 安装基础编译环境

进入 WSL Ubuntu：

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    git \
    cmake \
    ninja-build \
    clang-18 \
    llvm-18-dev \
    llvm-18-tools \
    libclang-18-dev \
    lld-18 \
    libzstd-dev
```

确认 LLVM 和 Clang：

```bash
clang++-18 --version
llvm-config --version
llvm-config --cmakedir
```

本项目使用 LLVM 18，因此 Enzyme 插件也必须针对 LLVM 18 编译。

## 3. 项目目录结构

推荐目录：

```text
TestEnzyme/
├── external/
│   ├── Enzyme/
│   ├── eigen/
│   ├── optim/
│   └── ClangEnzyme-18.so
│
├── projects/
│   └── 01TestAD/
│       ├── build.sh
│       ├── test.cpp
│       └── builds/
│
├── .gitignore
└── README.md
```

`external/` 用于存放第三方依赖和 Enzyme 插件。

## 4. 下载 Enzyme

在项目根目录：

```bash
mkdir -p external
cd external

git clone https://github.com/EnzymeAD/Enzyme.git
```

建议初始化所有 submodule：

```bash
cd Enzyme
git submodule update --init --recursive
```

## 5. 编译 ClangEnzyme 插件

首先确认 Clang 的 CMake 配置存在：

```bash
ls /usr/lib/llvm-18/lib/cmake/clang
```

然后：

```bash
cd ~/TestEnzyme/external/Enzyme

mkdir -p build
cd build

cmake -G Ninja ../enzyme \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
    -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
```

编译 Clang 插件：

```bash
ninja ClangEnzyme-18 -j$(nproc)
```

生成文件通常位于：

```text
external/Enzyme/build/Enzyme/ClangEnzyme-18.so
```

复制到 `external/`：

```bash
cp Enzyme/ClangEnzyme-18.so ../../ClangEnzyme-18.so
```

最终应有：

```text
external/ClangEnzyme-18.so
```

### bundle-includes.sh 问题

如果编译出现：

```text
bundle-includes.sh: Permission denied
```

执行：

```bash
chmod +x ~/TestEnzyme/external/Enzyme/enzyme/scripts/bundle-includes.sh
```

如果随后出现：

```text
bundle-includes.sh: not found
```

但文件实际存在，通常是 Windows CRLF 换行导致：

```bash
sed -i 's/\r$//' \
    ~/TestEnzyme/external/Enzyme/enzyme/scripts/bundle-includes.sh
```

然后重新执行 Ninja。

## 6. Enzyme 的几个 `.so`

Enzyme 构建后可能得到多个插件：

```text
ClangEnzyme-18.so
LLVMEnzyme-18.so
LLDEnzyme-18.so
```

本项目主要使用：

```text
ClangEnzyme-18.so
```

它可以直接通过 Clang 加载：

```bash
clang++-18 test.cpp \
    -O2 \
    -fplugin=../../external/ClangEnzyme-18.so \
    -o main
```

也就是说，普通 C++ 文件仍然直接交给 `clang++` 编译，只是在 Clang 编译过程中加载 Enzyme。

`LLVMEnzyme-18.so` 主要用于直接处理 LLVM IR，例如配合 `opt-18`。

`LLDEnzyme-18.so` 与 LLD 链接阶段相关。

当前 `build.sh` 只依赖：

```text
ClangEnzyme-18.so
```

## 7. 安装 Eigen

OptimLib 的 Eigen 后端要求 Eigen 3.4 或以上。

推荐直接使用 Eigen 3.4.0：

```bash
cd ~/TestEnzyme/external

git clone \
    --branch 3.4.0 \
    --depth 1 \
    https://gitlab.com/libeigen/eigen.git \
    eigen
```

目录应类似：

```text
external/eigen/Eigen/Dense
```

因此编译时 include 路径是：

```text
external/eigen
```

C++ 中：

```cpp
#include <Eigen/Dense>
```

## 8. 安装 OptimLib

```bash
cd ~/TestEnzyme/external

git clone https://github.com/kthohr/optim.git
```

然后初始化 OptimLib 使用的 submodule：

```bash
cd optim
git submodule update --init --recursive
```

目录应包含：

```text
external/optim/include/optim.hpp
```

编译时 include：

```text
external/optim/include
```

C++ 中：

```cpp
#include <optim.hpp>
```

如果使用 Eigen backend，需要定义：

```cpp
#define OPTIM_ENABLE_EIGEN_WRAPPERS
```

需要注意：

如果将 OptimLib 自己的 `.cpp` 文件作为独立编译单元，例如：

```text
external/optim/src/unconstrained/gd.cpp
```

那么只在 `test.cpp` 中定义该宏是不够的，因为宏不会传播到另一个编译单元。

这种情况下需要让编译器对所有 OptimLib 源文件定义：

```bash
-DOPTIM_ENABLE_EIGEN_WRAPPERS
```

或者单独构建 OptimLib。

## 9. build.sh

每个 project 可以拥有自己的 `build.sh`。

它主要负责：

- 指定需要编译的 `.c` / `.cpp`
- 指定第三方库 include 路径
- 自动定位项目根目录
- 自动定位 `external/`
- 自动加载 `ClangEnzyme-18.so`
- 使用 `clang++-18`
- 创建 `builds/`
- 将最终程序输出到 `builds/main`

典型结构：

```bash
#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXTERNAL="$ROOT/../../external"
BUILD="$ROOT/builds"

SOURCES=(
    "$ROOT/test.cpp"
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
    -O2
    -fplugin="$EXTERNAL/ClangEnzyme-18.so"
)

for dir in "${INCLUDES[@]}"; do
    CMD+=("-I$dir")
done

CMD+=("-o" "$BUILD/main")

"${CMD[@]}"

echo "Build success: $BUILD/main"
```

## 10. 添加新的 C/C++ 文件

只需要修改：

```bash
SOURCES=(
    "$ROOT/main.cpp"
    "$ROOT/dsp.cpp"
    "$ROOT/model.cpp"
)
```

也可以添加其他目录：

```bash
SOURCES=(
    "$ROOT/main.cpp"
    "$ROOT/src/dsp.cpp"
    "$ROOT/src/model.cpp"
)
```

## 11. 添加新的 header-only 库

假设：

```text
external/
├── eigen/
├── optim/
└── mylib/
    └── include/
```

只需要添加：

```bash
INCLUDES=(
    "$EXTERNAL/eigen"
    "$EXTERNAL/optim/include"
    "$EXTERNAL/mylib/include"
)
```

然后源码中即可正常：

```cpp
#include <mylib.hpp>
```

不需要写：

```cpp
#include "../../external/mylib/include/mylib.hpp"
```

## 12. 编译与运行

进入具体项目：

```bash
cd ~/TestEnzyme/projects/01TestAD
```

第一次给脚本执行权限：

```bash
chmod +x build.sh
```

编译：

```bash
./build.sh
```

运行：

```bash
./builds/main
```

## 13. Enzyme 版本匹配

Enzyme 是 LLVM 插件，因此 Enzyme 和 LLVM 的版本必须匹配。

本项目当前使用：

```text
LLVM 18
Clang 18
ClangEnzyme-18.so
```

对应的编译器为：

```bash
clang++-18
```

不要将针对其他 LLVM 版本构建的 Enzyme 插件直接加载到 Clang 18 中。

例如：

```text
Clang 18 + ClangEnzyme-18.so    OK

Clang 18 + ClangEnzyme-19.so    不应使用
```

如果未来升级到 LLVM 19，建议重新针对 LLVM 19 编译 Enzyme，并使用：

```text
clang++-19
ClangEnzyme-19.so
```