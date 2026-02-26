# swiftshaderCUDA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A fork of [SwiftShader](https://swiftshader.googlesource.com/SwiftShader) that offloads triangle traversal to a CUDA GPU while keeping pixel shading on the CPU. The Vulkan ICD API is unchanged — it is a drop-in replacement for the standard SwiftShader DLL.

## What it does

Standard SwiftShader rasterizes triangles entirely on the CPU: it iterates over every scanline pair of every primitive in software. This fork replaces that traversal step with a CUDA kernel:

1. **GPU** — the traversal kernel runs one thread-block per primitive. Each thread walks a stride of scanline pairs, computes 4-bit coverage masks for every 2×2 pixel quad, and appends covered quads to a work queue via atomic operations.
2. **CPU** — the work queue is downloaded to pinned host memory. The quads are distributed into 16 scanline-pair stripes and processed in parallel by marl worker tasks, each calling the same Reactor-JIT pixel routine (depth/stencil test, fragment shader, blending) that the unmodified renderer uses.

Pixel shading stays on the CPU so the existing JIT-compiled shaders work unmodified. Only the coverage traversal is accelerated.

## Architecture

```
processPixels()
│
├─ Upload Primitive span arrays to GPU (pre-allocated pinned pool, no cudaMalloc per frame)
│
├─ cudaLaunchTraversal()  ─────────────────────────────────────────────────────────────────
│   │  traverseKernel<<<primCount blocks, 128 threads>>>
│   │   • Each thread covers one scanline pair (y, y+1), strides by 256 rows
│   │   • Reads spans via __ldg() (read-only texture cache)
│   │   • Branchless 4-bit coverage mask per 2×2 quad
│   │   • Thread-local 512-entry quad buffer (one warp-level atomicAdd at flush)
│   └─ Output: flat QuadWorkItem[] + atomic count
│
├─ D2H memcpy: count, then work queue (pinned → host)
│
├─ Counting sort into 16 stripes by (y/2) % 16
│
└─ marl::WaitGroup — one task per non-empty stripe:
    • sort(primIdx, y, x)
    • merge consecutive full-coverage runs → one pixelRoutine call per scanline pair
    • per-quad: set outline spans, call JIT pixel routine
```

### Key data structures

| Type | Size | Description |
|------|------|-------------|
| `GPUSpan` | 4 B | `{uint16 left, uint16 right}` — one scanline's coverage extent |
| `GPUPrimitive` | variable | `{yMin, yMax, GPUSpan spans[yMax-yMin]}` — uploaded per primitive |
| `QuadWorkItem` | 8 B | `{int16 x, int16 y, uint32 cMask:4, uint32 primIdx:28}` |

Work queue capacity: 4 M quads × 8 B = **32 MB** device + 32 MB pinned host.

## Optimizations

| # | Location | Description |
|---|----------|-------------|
| 1 | CPU | Partial `memcpy` — copies only plane-equation bytes (~1800 B) when the primitive changes, skipping the outline array |
| 2 | CPU/GPU | Pre-allocated pools — no `cudaMalloc`/`cudaFree` per frame; all device and pinned buffers allocated once at init |
| 3 | CPU | 16-way marl parallelism — quads partitioned into 16 scanline-pair stripes, each processed by an independent marl worker |
| 4 | GPU | Thread-local accumulation buffer outside the y-loop — all scanline pairs of a thread accumulate together, minimising atomicAdd calls |
| 5 | GPU | Warp-level final flush — intra-warp inclusive prefix sum via `__shfl_up_sync` reduces 128 per-thread atomicAdds to 4 per block |
| 6 | CPU | Full-coverage run merging — consecutive `cMask=0xF` quads at the same `(primIdx, y)` merged into one `pixelRoutine` call, reducing calls from O(width/2) to O(1) per interior scanline pair |
| 7 | CPU | Odd-y guard fix — guard spans set to `{x, x}` instead of `{0, 0}` so the rasterizer starts at column `x`, not column 0 |
| 8 | GPU | `__ldg()` for span reads — routes loads through the read-only texture cache |
| 9 | CPU | Parallel stripe sorting — `std::sort` moved inside each marl task so all 16 stripe sorts execute in parallel rather than sequentially |
| 10 | CPU | Skip empty stripes — WaitGroup sized to non-empty stripes only; `marl::schedule` not called for zero-quad stripes |
| 11 | GPU | `__restrict__` on kernel pointer parameters — allows nvcc to prove no aliasing and elide redundant global-memory reloads |

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

- If a CUDA device is present, triangle traversal is GPU-accelerated for single-sample draw calls.
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

## Upstream SwiftShader

The sections below are from the original SwiftShader README and apply to this fork as well.

---

Introduction
------------

SwiftShader[^1] is a high-performance CPU-based implementation[^2] of the Vulkan[^3] 1.3 graphics API. Its goal is to provide hardware independence for advanced 3D graphics.

> NOTE: The [ANGLE](http://angleproject.org/) project can be used to achieve a layered implementation[^4] of OpenGL ES 3.1 (aka. "SwANGLE").

Testing
-------

SwiftShader's Vulkan implementation can be tested using the [dEQP](https://github.com/KhronosGroup/VK-GL-CTS) test suite.

See [docs/dEQP.md](docs/dEQP.md) for details.

Third-Party Dependencies
------------------------

The [third_party](third_party/) directory contains projects which originated outside of SwiftShader:

[subzero](third_party/subzero/) contains a fork of the [Subzero](https://chromium.googlesource.com/native_client/pnacl-subzero/) project. It originates from Google Chrome's (Portable) [Native Client](https://developer.chrome.com/native-client) project.

[llvm-subzero](third_party/llvm-subzero/) contains a minimized set of LLVM dependencies of the Subzero project.

[marl](third_party/marl/) is a hybrid thread/fibre task scheduler used for parallel pixel processing.

[googletest](third_party/googletest/) contains the [Google Test](https://github.com/google/googletest) project, as a Git submodule.

Documentation
-------------

See [docs/Index.md](docs/Index.md).

License
-------

The SwiftShader project is licensed under the Apache License Version 2.0. You can find a copy of it in [LICENSE.txt](LICENSE.txt).

Files in the third_party folder are subject to their respective license.

Authors and Contributors
------------------------

The legal authors for copyright purposes are listed in [AUTHORS.txt](AUTHORS.txt).

[CONTRIBUTORS.txt](CONTRIBUTORS.txt) contains a list of names of individuals who have contributed to SwiftShader.

Notes and Disclaimers
---------------------

[^1]: This is not an official Google product.
[^2]: Vulkan 1.3 conformance: https://www.khronos.org/conformance/adopters/conformant-products#submission_717
[^3]: Trademarks are the property of their respective owners.
[^4]: OpenGL ES 3.1 conformance: https://www.khronos.org/conformance/adopters/conformant-products/opengles#submission_906
