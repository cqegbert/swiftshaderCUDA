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

// CUDA kernel for triangle traversal.
//
// Each thread block handles one primitive.  Threads within the block stride
// through scanline pairs, computing coverage masks for each 2×2 pixel quad
// and appending covered quads to a work queue via atomicAdd.
//
// The resulting work queue is downloaded to pinned CPU memory by
// CudaRasterizer::processPixels(), which then invokes the existing
// Reactor-JIT pixel routine for each quad.

#include "CudaTraversal.cuh"

// Undefine macros that conflict with CUDA's library_types.h enum member names.
// SwiftShader's Version.hpp (or headers it pulls in) defines MAJOR_VERSION,
// MINOR_VERSION, and PATCH_LEVEL as plain integers, which breaks the CUDA
// library_types.h enum that uses those identifiers.
#ifdef MAJOR_VERSION
#  undef MAJOR_VERSION
#endif
#ifdef MINOR_VERSION
#  undef MINOR_VERSION
#endif
#ifdef PATCH_LEVEL
#  undef PATCH_LEVEL
#endif

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

// Number of threads per block.  Must be a multiple of 32 (warp size).
// Each thread handles one scanline pair.  128 threads covers 256 scanlines
// per block iteration, which is enough for typical triangle heights.
static constexpr int THREADS_PER_BLOCK = 128;

// Kernel: one block per primitive, threads stride through scanline pairs.
__global__ void traverseKernel(
    sw::GPUPrimitive **primPtrs,
    int primCount,
    sw::QuadWorkItem *queue,
    int *count,
    int maxItems)
{
	const int primIdx = blockIdx.x;
	if(primIdx >= primCount) return;

	const sw::GPUPrimitive *p = primPtrs[primIdx];
	if(p == nullptr) return;  // degenerate primitive (spanCount <= 0), skip safely
	const int yMin = p->yMin;
	const int yMax = p->yMax;

	// Each thread handles one scanline pair (y, y+1).
	// Threads stride by blockDim.x * 2 through the primitive's height.
	for(int y = yMin + static_cast<int>(threadIdx.x) * 2; y < yMax; y += THREADS_PER_BLOCK * 2)
	{
		const int spanIdx0 = y - yMin;
		const int spanIdx1 = spanIdx0 + 1;

		const sw::GPUSpan sy  = p->spans[spanIdx0];
		const sw::GPUSpan sy1 = p->spans[spanIdx1];

		// Union of both rows' spans (aligned to even x for 2×2 quads).
		const int x0 = (min(static_cast<int>(sy.left), static_cast<int>(sy1.left))) & ~1;
		const int x1 =  max(static_cast<int>(sy.right), static_cast<int>(sy1.right));

		for(int x = x0; x < x1; x += 2)
		{
			// Compute 4-bit coverage mask for this 2×2 quad.
			// bit 0: pixel (x,   y)   bit 1: pixel (x+1, y)
			// bit 2: pixel (x,   y+1) bit 3: pixel (x+1, y+1)
			uint8_t cMask = 0;
			if(x     >= sy.left  && x     < sy.right)  cMask |= 1;
			if(x + 1 >= sy.left  && x + 1 < sy.right)  cMask |= 2;
			if(x     >= sy1.left && x     < sy1.right)  cMask |= 4;
			if(x + 1 >= sy1.left && x + 1 < sy1.right)  cMask |= 8;

			if(cMask)
			{
				const int idx = atomicAdd(count, 1);
				if(idx < maxItems)
				{
					sw::QuadWorkItem item;
					item.x       = static_cast<int16_t>(x);
					item.y       = static_cast<int16_t>(y);
					item.cMask   = static_cast<uint32_t>(cMask);
					item.primIdx = static_cast<uint32_t>(primIdx);
					queue[idx]   = item;
				}
			}
		}
	}
}

}  // namespace

// Packs each primitive's outline spans into a contiguous device buffer,
// then launches the traversal kernel.
extern "C" void cudaLaunchTraversal(
    sw::GPUPrimitive **d_primPtrs,
    int primCount,
    sw::QuadWorkItem *d_queue,
    int *d_count,
    int maxItems,
    void *stream)
{
	if(primCount <= 0) return;

	const cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);

	// One block per primitive; each block uses THREADS_PER_BLOCK threads.
	const dim3 grid(static_cast<unsigned int>(primCount));
	const dim3 block(THREADS_PER_BLOCK);

	traverseKernel<<<grid, block, 0, cudaStream>>>(
	    d_primPtrs, primCount, d_queue, d_count, maxItems);
}
