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

// Maximum quads a single thread accumulates before a per-thread overflow flush.
// 512 entries × 8 bytes = 4 KB per thread in CUDA local memory (DRAM-backed;
// L2 may cache recently written entries but this does not reside in registers).
// The buffer lives OUTSIDE the scanline-pair loop so small/medium triangles
// pay only one warp-level atomicAdd per warp total rather than one per thread.
// When the buffer fills mid-loop an overflow flush is issued (one per-thread
// atomicAdd per kLocalBufCap quads), keeping atomicAdd count bounded.
static constexpr int kLocalBufCap = 512;

// ---------------------------------------------------------------------------
// Opt 5: warp-level final flush.
// ---------------------------------------------------------------------------
//
// After all threads in the warp have exited the y-loop, they converge at this
// function.  Instead of 32 separate atomicAdd calls (one per lane), we:
//
//   1. Compute an intra-warp inclusive prefix sum of localCount by shuffling
//      the RUNNING SUM 'val' (not the original localCount) via __shfl_up_sync.
//      The exclusive offset for each lane is val - localCount.
//   2. Lane 0 issues a single atomicAdd(count, warpTotal) to claim the slice.
//   3. Each lane writes its items to queue[warpBase + myOffset + k].
//
// IMPORTANT: shuffling the original localCount (not the running sum) would
// give wrong results for non-power-of-2 inter-lane distances — e.g. lane 3
// can't see lane 0 directly.  Shuffling 'val' propagates partial sums so
// every lane correctly accumulates all predecessors.
//
// Result: THREADS_PER_BLOCK/32 = 4 atomicAdds per block (down from 128).
// This is the common path for small/medium triangles that never overflow.
// Large triangles also benefit on their final (partial) buffer.
//
// Correctness requirement: all 32 lanes must be at the same program counter
// when this function is called (no active divergence within the warp).
// This is guaranteed because the function is called after the y-loop, where
// every lane has exited and the warp has fully re-converged.
__device__ __forceinline__ void flushWarp(
    const sw::QuadWorkItem *localBuf,
    int                     localCount,
    sw::QuadWorkItem       *queue,
    int                    *count,
    int                     maxItems)
{
	static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
	const int laneId = static_cast<int>(threadIdx.x) & 31;

	// Inclusive prefix sum: evolve 'val' from per-lane count to running total.
	// Each iteration propagates partial sums across twice the previous distance,
	// so after log2(32)=5 steps val[i] = sum(localCount[0..i]).
	int val = localCount;
	for(int delta = 1; delta < 32; delta <<= 1)
	{
		const int n = __shfl_up_sync(FULL_MASK, val, static_cast<unsigned>(delta));
		if(laneId >= delta) val += n;
	}

	// Exclusive prefix sum = inclusive sum − own count.
	const int myOffset  = val - localCount;
	// Warp total = val[31] = inclusive prefix sum of all 32 lanes.
	const int warpTotal = __shfl_sync(FULL_MASK, val, 31);
	if(warpTotal == 0) return;  // Whole warp has nothing to flush (no divergence).

	// Lane 0 claims a contiguous slot for the entire warp.
	int warpBase = 0;
	if(laneId == 0)
		warpBase = atomicAdd(count, warpTotal);

	// Broadcast the claimed base to all lanes.
	warpBase = __shfl_sync(FULL_MASK, warpBase, 0);

	// Each lane writes its items at its exclusive-prefix-sum offset.
	const int writeBase = warpBase + myOffset;
	for(int k = 0; k < localCount; ++k)
	{
		const int idx = writeBase + k;
		if(idx < maxItems) queue[idx] = localBuf[k];
	}
}

// Kernel: one block per primitive, threads stride through scanline pairs.
// Opt 4 (enhanced): thread-local buffer lives outside the y-loop so quads
// from all scanline pairs accumulate together, minimising total atomicAdds.
// Opt 5: final flush uses warp-level aggregation (4 atomicAdds for 128
// threads) instead of per-thread atomicAdds (128).  Overflow flushes that
// occur mid-loop when the buffer is full remain per-thread (rare in practice).
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

	// Thread-local accumulation buffer — outside the y-loop so it persists
	// across scanline pairs, amortising atomicAdd cost over all quads.
	sw::QuadWorkItem localBuf[kLocalBufCap];
	int localCount = 0;

	// Each thread handles one scanline pair (y, y+1).
	// Threads stride by blockDim.x * 2 through the primitive's height.
	for(int y = yMin + static_cast<int>(threadIdx.x) * 2; y < yMax; y += THREADS_PER_BLOCK * 2)
	{
		const int spanIdx0 = y - yMin;
		const int spanIdx1 = spanIdx0 + 1;

		// Load both spans as 32-bit words to avoid repeated uint16_t casts.
		// GPUSpan layout: { uint16_t left; uint16_t right } → little-endian:
		//   bits [0..15]  = left,  bits [16..31] = right
		// Opt 8: __ldg() routes reads through the read-only (texture) cache,
		// keeping L1 available for write-back data and improving throughput when
		// span arrays are shared across multiple warp iterations.
		const uint32_t raw0 = __ldg(reinterpret_cast<const uint32_t *>(&p->spans[spanIdx0]));
		const uint32_t raw1 = __ldg(reinterpret_cast<const uint32_t *>(&p->spans[spanIdx1]));
		const int l0 = (int)(raw0 & 0xFFFFu);
		const int r0 = (int)(raw0 >> 16);
		const int l1 = (int)(raw1 & 0xFFFFu);
		const int r1 = (int)(raw1 >> 16);

		// Union of both rows' spans (aligned to even x for 2×2 quads).
		const int x0 = min(l0, l1) & ~1;
		const int x1 = max(r0, r1);

		for(int x = x0; x < x1; x += 2)
		{
			// Compute 4-bit coverage mask for this 2×2 quad — branchless.
			// bit 0: pixel (x,   y)   bit 1: pixel (x+1, y)
			// bit 2: pixel (x,   y+1) bit 3: pixel (x+1, y+1)
			// Bitwise & instead of && avoids branch divergence across the warp.
			const uint32_t cMask =
			    ((uint32_t)((x     >= l0) & (x     < r0))      ) |
			    ((uint32_t)((x + 1 >= l0) & (x + 1 < r0)) << 1 ) |
			    ((uint32_t)((x     >= l1) & (x     < r1)) << 2 ) |
			    ((uint32_t)((x + 1 >= l1) & (x + 1 < r1)) << 3 );

			if(cMask)
			{
				if(localCount == kLocalBufCap)
				{
					// Buffer full: overflow flush with one per-thread atomicAdd.
					// Rare: only occurs when one thread processes > kLocalBufCap quads
					// across all its scanline pairs (very wide or very tall triangles).
					const int base = atomicAdd(count, localCount);
					for(int k = 0; k < localCount; ++k)
					{
						const int idx = base + k;
						if(idx < maxItems) queue[idx] = localBuf[k];
					}
					localCount = 0;
				}
				localBuf[localCount].x       = static_cast<int16_t>(x);
				localBuf[localCount].y       = static_cast<int16_t>(y);
				localBuf[localCount].cMask   = cMask;
				localBuf[localCount].primIdx = static_cast<uint32_t>(primIdx);
				++localCount;
			}
		}
	}

	// Final flush: warp-level aggregation — THREADS_PER_BLOCK/32 = 4 atomicAdds
	// for a 128-thread block (vs 128 with per-thread atomicAdd).
	// All warp members have exited the y-loop and are at the same PC here.
	flushWarp(localBuf, localCount, queue, count, maxItems);
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
