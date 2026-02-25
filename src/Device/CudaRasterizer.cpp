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

#include "CudaRasterizer.hpp"
#include "CudaTraversal.cuh"

#include "System/Debug.hpp"

// CudaRasterizer.hpp → Renderer.hpp → VkConfig.hpp → Version.hpp defines
// MAJOR_VERSION and MINOR_VERSION as plain integer macros.  CUDA's
// library_types.h uses those exact identifiers as enum members, so including
// cuda_runtime.h with those macros live causes a syntax error.  Undef them
// here, right before pulling in the CUDA runtime, then redefine them below.
#ifdef MAJOR_VERSION
#  define _SW_SAVED_MAJOR_VERSION MAJOR_VERSION
#  undef  MAJOR_VERSION
#endif
#ifdef MINOR_VERSION
#  define _SW_SAVED_MINOR_VERSION MINOR_VERSION
#  undef  MINOR_VERSION
#endif

#include <cuda_runtime.h>

// Restore the SwiftShader version macros so downstream code still compiles.
#ifdef _SW_SAVED_MAJOR_VERSION
#  define MAJOR_VERSION _SW_SAVED_MAJOR_VERSION
#  undef  _SW_SAVED_MAJOR_VERSION
#endif
#ifdef _SW_SAVED_MINOR_VERSION
#  define MINOR_VERSION _SW_SAVED_MINOR_VERSION
#  undef  _SW_SAVED_MINOR_VERSION
#endif

#include "marl/defer.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <mutex>

// ---------------------------------------------------------------------------
// Macro helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                       \
	do {                                                                       \
		cudaError_t _err = (expr);                                             \
		if(_err != cudaSuccess)                                                \
		{                                                                      \
			sw::warn("CUDA error %d (%s) at %s:%d\n",                         \
			         static_cast<int>(_err), cudaGetErrorString(_err),         \
			         __FILE__, __LINE__);                                       \
		}                                                                      \
	} while(0)

namespace sw {

// ---------------------------------------------------------------------------
// Module-level state (initialised once)
// ---------------------------------------------------------------------------

namespace {

std::once_flag g_initFlag;
bool g_available = false;

// CUDA stream used for all async operations.
cudaStream_t g_stream = nullptr;

// Pinned host memory for uploading span data and downloading work items.
static constexpr int kMaxWorkItems = CUDA_MAX_WORK_ITEMS;
static constexpr int kMaxPrims     = sw::MaxBatchSize;

// Number of parallel scanline-pair stripes (= MaxClusterCount = 16).
static constexpr int kNumStripes = sw::MaxClusterCount;

// Per-quad cost reduction: copy only plane-equation bytes, skip full memcpy.
// offsetof(Primitive, outline) == 1800 bytes on all supported platforms.
static constexpr size_t kPrimEquationBytes = offsetof(sw::Primitive, outline);

// Fixed slot size for one primitive's GPU span data.
// spans[0] is already in sizeof(GPUPrimitive); add OUTLINE_RESOLUTION more
// slots plus one guard span = OUTLINE_RESOLUTION+1 additional spans.
static constexpr size_t kPrimSlotSize =
    sizeof(GPUPrimitive) + (OUTLINE_RESOLUTION + 1) * sizeof(GPUSpan);  // ~32784 bytes

QuadWorkItem *g_pinnedWorkQueue = nullptr;  // host-side, pinned
int          *g_pinnedCount     = nullptr;  // host-side atomic count, pinned

// Second work-queue buffer for sorted output (pinned).
QuadWorkItem *g_sortedQueue = nullptr;

// Device-side work queue and atomic counter.
QuadWorkItem *g_devWorkQueue = nullptr;
int          *g_devCount     = nullptr;

// Device-side per-primitive span buffers and pointer array.
// All device slabs live inside g_devPrimPool; pointers are set once at init.
GPUPrimitive *g_devPrimSlabs[kMaxPrims] = {};  // fixed offsets into pool
GPUPrimitive **g_devPrimPtrs            = nullptr;  // device array of pointers
GPUPrimitive **g_pinnedPrimPtrs         = nullptr;  // pinned host array of pointers

// Pre-allocated pools (Opt 2): no per-frame cudaMalloc/cudaFree.
GPUPrimitive *g_devPrimPool    = nullptr;  // device, kMaxPrims * kPrimSlotSize
char         *g_pinnedPrimPool = nullptr;  // pinned host, same size

void initOnce()
{
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if(err != cudaSuccess || deviceCount == 0)
	{
		g_available = false;
		return;
	}

	g_available = true;

	CUDA_CHECK(cudaStreamCreate(&g_stream));

	// Work queues (primary + sorted)
	CUDA_CHECK(cudaMallocHost(&g_pinnedWorkQueue, sizeof(QuadWorkItem) * kMaxWorkItems));
	CUDA_CHECK(cudaMallocHost(&g_pinnedCount,     sizeof(int)));
	CUDA_CHECK(cudaMallocHost(&g_sortedQueue,     sizeof(QuadWorkItem) * kMaxWorkItems));
	CUDA_CHECK(cudaMalloc(&g_devWorkQueue,        sizeof(QuadWorkItem) * kMaxWorkItems));
	CUDA_CHECK(cudaMalloc(&g_devCount,            sizeof(int)));

	// Primitive pointer arrays
	CUDA_CHECK(cudaMallocHost(&g_pinnedPrimPtrs, sizeof(GPUPrimitive *) * kMaxPrims));
	CUDA_CHECK(cudaMalloc(&g_devPrimPtrs,        sizeof(GPUPrimitive *) * kMaxPrims));

	// Pre-allocate fixed-size span pools (Opt 2): one allocation each, sliced.
	CUDA_CHECK(cudaMalloc    (&g_devPrimPool,    (size_t)kMaxPrims * kPrimSlotSize));
	CUDA_CHECK(cudaMallocHost(&g_pinnedPrimPool, (size_t)kMaxPrims * kPrimSlotSize));

	// Point each slab pointer to its fixed slot — never changes after init.
	for(int i = 0; i < kMaxPrims; ++i)
		g_devPrimSlabs[i] = reinterpret_cast<GPUPrimitive *>(
		    reinterpret_cast<char *>(g_devPrimPool) + i * kPrimSlotSize);
}

}  // namespace

// ---------------------------------------------------------------------------
// CudaRasterizer public API
// ---------------------------------------------------------------------------

bool CudaRasterizer::isAvailable()
{
	std::call_once(g_initFlag, initOnce);
	return g_available;
}

void CudaRasterizer::init()
{
	(void)isAvailable();  // triggers initOnce
}

void CudaRasterizer::shutdown()
{
	if(!g_available) return;

	cudaStreamSynchronize(g_stream);
	cudaStreamDestroy(g_stream);

	cudaFreeHost(g_pinnedWorkQueue);
	cudaFreeHost(g_pinnedCount);
	cudaFreeHost(g_sortedQueue);
	cudaFree(g_devWorkQueue);
	cudaFree(g_devCount);

	cudaFreeHost(g_pinnedPrimPtrs);
	cudaFree(g_devPrimPtrs);

	// Pool frees — individual g_devPrimSlabs[i] are offsets into the pool.
	cudaFree    (g_devPrimPool);
	cudaFreeHost(g_pinnedPrimPool);

	g_available = false;
}

// ---------------------------------------------------------------------------
// processPixels
// ---------------------------------------------------------------------------

void CudaRasterizer::processPixels(
    const Primitive *primitives,
    int count,
    PixelProcessor::RoutineType pixelRoutine,
    DrawData *data,
    vk::Device *device)
{
	if(count <= 0) return;

	// ------------------------------------------------------------------
	// 1. Upload primitive outline spans to GPU (Opt 2: no per-prim alloc)
	// ------------------------------------------------------------------
	for(int i = 0; i < count; ++i)
	{
		const Primitive &p = primitives[i];
		const int spanCount = p.yMax - p.yMin;
		if(spanCount <= 0)
		{
			g_pinnedPrimPtrs[i] = nullptr;
			continue;
		}

		// Size of actual data to transfer (spanCount entries + one guard span).
		const size_t slab = sizeof(GPUPrimitive) + sizeof(GPUSpan) * spanCount;

		// Use pre-allocated pinned slot for staging — no heap alloc (Opt 2).
		GPUPrimitive *hp = reinterpret_cast<GPUPrimitive *>(
		    g_pinnedPrimPool + (size_t)i * kPrimSlotSize);
		hp->yMin = p.yMin;
		hp->yMax = p.yMax;
		for(int s = 0; s < spanCount; ++s)
		{
			hp->spans[s].left  = p.outline[p.yMin + s].left;
			hp->spans[s].right = p.outline[p.yMin + s].right;
		}
		// Guard span: kernel may read spans[spanCount] when spanCount is odd.
		hp->spans[spanCount] = { 0, 0 };

		// Upload only the actual slab bytes (not the full fixed slot).
		CUDA_CHECK(cudaMemcpyAsync(g_devPrimSlabs[i], hp, slab,
		                           cudaMemcpyHostToDevice, g_stream));
		g_pinnedPrimPtrs[i] = g_devPrimSlabs[i];
	}

	// Upload pointer array to device.
	CUDA_CHECK(cudaMemcpyAsync(g_devPrimPtrs, g_pinnedPrimPtrs,
	                           sizeof(GPUPrimitive *) * count,
	                           cudaMemcpyHostToDevice, g_stream));

	// ------------------------------------------------------------------
	// 2. Zero the work-queue counter on the device
	// ------------------------------------------------------------------
	CUDA_CHECK(cudaMemsetAsync(g_devCount, 0, sizeof(int), g_stream));

	// ------------------------------------------------------------------
	// 3. Launch traversal kernel
	// ------------------------------------------------------------------
	cudaLaunchTraversal(g_devPrimPtrs, count,
	                    g_devWorkQueue, g_devCount,
	                    kMaxWorkItems,
	                    static_cast<void *>(g_stream));

	// ------------------------------------------------------------------
	// 4. Download work queue and count (synchronous – waits for kernel)
	// ------------------------------------------------------------------
	CUDA_CHECK(cudaMemcpyAsync(g_pinnedCount, g_devCount, sizeof(int),
	                           cudaMemcpyDeviceToHost, g_stream));
	CUDA_CHECK(cudaStreamSynchronize(g_stream));  // wait for count

	const int numQuads = *g_pinnedCount;
	if(numQuads <= 0) return;

	const int clampedQuads = (numQuads < kMaxWorkItems) ? numQuads : kMaxWorkItems;

	CUDA_CHECK(cudaMemcpyAsync(g_pinnedWorkQueue, g_devWorkQueue,
	                           sizeof(QuadWorkItem) * clampedQuads,
	                           cudaMemcpyDeviceToHost, g_stream));
	CUDA_CHECK(cudaStreamSynchronize(g_stream));  // wait for data

	// ------------------------------------------------------------------
	// 5. Sort quads into 16 scanline-pair stripes, then process in parallel
	//    (Opt 1: partial memcpy only; Opt 3: 16-way marl parallelism)
	//
	// Thread-safety: worker t handles quads where (item.y/2) % 16 == t.
	// Each 2×2 quad occupies a unique scanline pair, so workers never
	// write to the same pixel row — no locks needed.
	// ------------------------------------------------------------------

	// Pass 1: count quads per stripe (O(N) counting sort).
	int stripeCounts[kNumStripes] = {};
	for(int q = 0; q < clampedQuads; ++q)
		stripeCounts[(g_pinnedWorkQueue[q].y / 2) % kNumStripes]++;

	// Exclusive prefix sum → stripe start offsets.
	int stripeStarts[kNumStripes + 1];
	stripeStarts[0] = 0;
	for(int i = 0; i < kNumStripes; ++i)
		stripeStarts[i + 1] = stripeStarts[i] + stripeCounts[i];

	// Pass 2: scatter into sorted buffer.
	int pos[kNumStripes];
	std::copy(stripeStarts, stripeStarts + kNumStripes, pos);
	for(int q = 0; q < clampedQuads; ++q)
	{
		int s = (g_pinnedWorkQueue[q].y / 2) % kNumStripes;
		g_sortedQueue[pos[s]++] = g_pinnedWorkQueue[q];
	}

	// Sort each stripe by (primIdx, y).
	// Sorting primIdx-first maximises cache hits in the worker loop: the
	// lastPrimIdx optimisation copies kPrimEquationBytes (1800 B) only when
	// the primitive changes, so grouping all quads from the same primitive
	// together reduces copies from O(N) to O(distinct_primitives_per_stripe).
	// Within a primitive, sorting by y preserves raster-order for blending.
	for(int s = 0; s < kNumStripes; ++s)
		std::sort(g_sortedQueue + stripeStarts[s],
		          g_sortedQueue + stripeStarts[s + 1],
		          [](const QuadWorkItem &a, const QuadWorkItem &b) {
			          return a.primIdx != b.primIdx ? a.primIdx < b.primIdx : a.y < b.y;
		          });

	// Dispatch one marl task per stripe.
	marl::WaitGroup wg(kNumStripes);
	for(int s = 0; s < kNumStripes; ++s)
	{
		const int qStart = stripeStarts[s];
		const int qEnd   = stripeStarts[s + 1];

		marl::schedule([=, &wg]() {
			defer(wg.done());

			// Each worker has its own stack-local Primitive — no sharing.
			Primitive local;
			int lastPrimIdx = -1;

			for(int q = qStart; q < qEnd; ++q)
			{
				const QuadWorkItem &item = g_sortedQueue[q];
				const int x             = item.x;
				const int y             = item.y;
				const uint8_t cMask     = static_cast<uint8_t>(item.cMask);

				// Opt 1: copy only plane-equation bytes when primitive changes.
				if(static_cast<int>(item.primIdx) != lastPrimIdx)
				{
					std::memcpy(&local, &primitives[item.primIdx], kPrimEquationBytes);
					lastPrimIdx = static_cast<int>(item.primIdx);
				}

				local.yMin = y;
				local.yMax = y + 2;

				// QuadRasterizer::generate() aligns yMin &= -2, rounding any ODD y
				// down to y-1.  That produces TWO rasterizer loop iterations:
				//   iter 0: reads outline[y-1] and outline[y]
				//   iter 1: reads outline[y+1] and outline[y+2]
				// The entries at y-1 and y+2 are outside our coverage range and may
				// contain stale values from earlier quads processed in this worker's
				// stripe.  Zero them so both extra rows produce empty spans and the
				// rasterizer writes no pixels there.
				if(y & 1)
				{
					local.outline[y - 1] = { 0, 0 };
					local.outline[y + 2] = { 0, 0 };
				}

				// Row y (bits 0, 1)
				local.outline[y].left   = static_cast<uint16_t>((cMask & 1) ? x     : ((cMask & 2) ? x + 1 : x + 2));
				local.outline[y].right  = static_cast<uint16_t>((cMask & 2) ? x + 2 : ((cMask & 1) ? x + 1 : x    ));
				// Row y+1 (bits 2, 3)
				local.outline[y + 1].left  = static_cast<uint16_t>((cMask & 4) ? x     : ((cMask & 8) ? x + 1 : x + 2));
				local.outline[y + 1].right = static_cast<uint16_t>((cMask & 8) ? x + 2 : ((cMask & 4) ? x + 1 : x    ));

				pixelRoutine(device, &local, 1, 0, 1, data);
			}
		});
	}
	wg.wait();
}

}  // namespace sw
