// Copyright 2024 The SwiftShader Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared data structures for CUDA-accelerated quad traversal.
// This header is included by both .cu (device) and .cpp (host) files.
// It intentionally avoids any CUDA-specific headers so it can be included
// from plain C++ without the CUDA toolkit on the include path.

#ifndef sw_CudaTraversal_cuh
#define sw_CudaTraversal_cuh

#include <cstdint>

namespace sw {

// Compact representation of one scanline span for the GPU.
// Mirrors Primitive::Span but without the surrounding Primitive overhead.
struct GPUSpan
{
	uint16_t left;
	uint16_t right;
};

// Per-primitive data uploaded to the GPU.
// Variable-length: spans[] contains (yMax - yMin) entries covering [yMin, yMax).
// Allocated on the device with cudaMalloc.
struct GPUPrimitive
{
	int yMin;
	int yMax;
	// spans[i] corresponds to scanline (yMin + i)
	GPUSpan spans[1];  // flexible array member; actual length = (yMax - yMin)
};

// One entry in the work queue output by the traversal kernel.
// Describes a 2×2 pixel quad that is (partially) covered by a triangle.
//
// Layout (8 bytes, no padding):
//   [0..1] x       – left column (always even)
//   [2..3] y       – top row (always even)
//   [4..7] cMask:4 – 4-bit coverage mask (bits 0..3)
//          primIdx:28 – primitive index (bits 4..31)
struct QuadWorkItem
{
	int16_t  x;          // Left column of the quad (always even)
	int16_t  y;          // Top row of the quad (always even)
	uint32_t cMask   :  4;  // 4-bit coverage: bit0=(x,y) bit1=(x+1,y) bit2=(x,y+1) bit3=(x+1,y+1)
	uint32_t primIdx : 28;  // Index into the batch's Primitive array
};

static_assert(sizeof(GPUSpan) == 4, "GPUSpan size mismatch");
static_assert(sizeof(QuadWorkItem) == 8, "QuadWorkItem size mismatch");

}  // namespace sw

// Maximum number of QuadWorkItems per batch (32 MB of device memory).
static constexpr int CUDA_MAX_WORK_ITEMS = 4 * 1024 * 1024;

// C-linkage launcher called from CudaRasterizer.cpp.
// d_primPtrs: device pointer to array of (primCount) GPUPrimitive* pointers
// d_queue:    device pointer to pre-allocated QuadWorkItem array
// d_count:    device pointer to atomic int counter (must be zeroed before call)
// maxItems:   capacity of d_queue (typically CUDA_MAX_WORK_ITEMS)
// stream:     CUDA stream to use (may be 0 for default stream)
extern "C" void cudaLaunchTraversal(
    sw::GPUPrimitive **d_primPtrs,
    int primCount,
    sw::QuadWorkItem *d_queue,
    int *d_count,
    int maxItems,
    void *stream);  // typed as void* to avoid pulling in cuda_runtime.h here

#endif  // sw_CudaTraversal_cuh
