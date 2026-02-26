# swiftshaderCUDA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A fork of [SwiftShader](https://swiftshader.googlesource.com/SwiftShader) that leverages CUDA for accelerating graphics work via a GPU. The motivation behind this is to enable GPUs which do not support Vulkan natively, such as the NVIDIA A100, to perform GPU accelerated graphics work.  I expect this to be relevant for robotics training scenarios in particular, where a cluster of GPUS that are being used for ML training could additionally run graphics simulations as part of their training.

The Vulkan ICD API is unchanged — it is a drop-in replacement for the standard SwiftShader DLL.


## Requirements

| Component | Minimum version |
|-----------|----------------|
| Windows | 10 / 11 (x64) |
| Visual Studio | 2022 or 2026 (MSVC 19.30+) — **x64 host required** |
| CMake | 3.26+ |
| Ninja | any recent version |
| CUDA Toolkit | 12.x or 13.x |
| NVIDIA GPU | Compute capability 7.5+ (Turing / RTX 20 series or newer) |

> **Important:** always initialise the x64 MSVC environment before running CMake (`vcvarsall.bat x64`). Using the x86 host causes a `cudafe++` ACCESS_VIOLATION during CUDA language detection.

Supported CUDA compute architectures: **7.5, 8.0, 8.6, 8.9, 9.0** (Turing, Ampere, Ada Lovelace, Hopper).

If no CUDA-capable device or toolkit is detected, CMake skips the CUDA sources and the build falls back to standard SwiftShader automatically.

## Building on Windows

The repository includes a ready-made build script for Windows:

```bat
build_dll.bat
```

This script:
1. Calls `vcvarsall.bat x64` to set up the x64 MSVC environment.
2. Configures the project with Ninja and the required CMake flags.
3. Builds the `vk_swiftshader` target.

Output: `out\build\x64-Release\vk_swiftshader.dll`

### Manual CMake invocation

From a developer command prompt with the **x64** MSVC environment active:

```bat
cmake -G Ninja ^
      -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
      -DSWIFTSHADER_BUILD_TESTS=OFF ^
      -DSWIFTSHADER_BUILD_BENCHMARKS=OFF ^
      -S . -B out\build\x64-Release

cmake --build out\build\x64-Release --target vk_swiftshader
```

> `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required because the marl submodule uses a CMake version older than 3.5 and CMake 4.3+ changed the minimum policy handling.

### Verifying CUDA detection

During CMake configuration, look for:

```
-- Found CUDA: <path>  (found version "13.x")
-- CUDA architectures: 75;80;86;89;90
```

If CUDA is not found, the build still succeeds and produces a CPU-only DLL.

## Usage

swiftshaderCUDA is a drop-in replacement for the standard SwiftShader ICD:

**Option A — place alongside the application executable:**

Copy `vk_swiftshader.dll` and `vk_swiftshader_icd.json` next to your application's `.exe`. Most Vulkan loaders search the application directory first.

**Option B — set the ICD environment variable:**

```bat
set VK_ICD_FILENAMES=C:\path\to\vk_swiftshader_icd.json
your_vulkan_app.exe
```

**Option C — rename and replace `vulkan-1.dll`:**

Some applications load `vulkan-1.dll` directly rather than going through the loader. Renaming the DLL allows it to be used as a direct replacement.

### Runtime behaviour

- If a CUDA device is present, graphics work is GPU-accelerated for single-sample draw calls.
- Multi-sample (MSAA) draw calls fall back to the standard 16-cluster CPU rasterization path.
- CUDA initialisation is lazy — the first draw call triggers it.
- If CUDA fails at runtime (driver mismatch, OOM, etc.) the affected draw call is silently skipped; the CPU path is not automatically used as fallback for those quads.

## Source layout

```
src/Device/
├── CudaTraversal.cuh      # Shared header: GPUSpan, GPUPrimitive, QuadWorkItem
├── CudaTraversal.cu       # CUDA traversal kernel + cudaLaunchTraversal()
├── CudaRasterizer.hpp     # Public API: isAvailable(), init(), shutdown(), processPixels()
└── CudaRasterizer.cpp     # CPU-side: upload, sort, merge, marl dispatch
```

The CUDA sources are compiled into the `vk_device` static library and linked into `vk_swiftshader`. The `SWIFTSHADER_ENABLE_CUDA` preprocessor define guards all CUDA-specific code in the renderer.

---
