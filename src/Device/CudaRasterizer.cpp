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

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

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
// Reused across frames; grown on demand.
static constexpr int kMaxWorkItems = CUDA_MAX_WORK_ITEMS;
static constexpr int kMaxPrims     = sw::MaxBatchSize;

QuadWorkItem *g_pinnedWorkQueue = nullptr;  // host-side, pinned
int          *g_pinnedCount     = nullptr;  // host-side atomic count, pinned

// Device-side work queue and atomic counter.
QuadWorkItem *g_devWorkQueue = nullptr;
int          *g_devCount     = nullptr;

// Device-side per-primitive span buffers and pointer array.
// We keep one large slab and suballocate per batch.
GPUPrimitive *g_devPrimSlabs[kMaxPrims] = {};  // per-primitive device buffers
GPUPrimitive **g_devPrimPtrs            = nullptr;  // device array of pointers
GPUPrimitive **g_pinnedPrimPtrs         = nullptr;  // pinned host array of pointers

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

	// Work queue
	CUDA_CHECK(cudaMallocHost(&g_pinnedWorkQueue, sizeof(QuadWorkItem) * kMaxWorkItems));
	CUDA_CHECK(cudaMallocHost(&g_pinnedCount,     sizeof(int)));
	CUDA_CHECK(cudaMalloc(&g_devWorkQueue,        sizeof(QuadWorkItem) * kMaxWorkItems));
	CUDA_CHECK(cudaMalloc(&g_devCount,            sizeof(int)));

	// Primitive pointer arrays
	CUDA_CHECK(cudaMallocHost(&g_pinnedPrimPtrs, sizeof(GPUPrimitive *) * kMaxPrims));
	CUDA_CHECK(cudaMalloc(&g_devPrimPtrs,        sizeof(GPUPrimitive *) * kMaxPrims));
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
	cudaFree(g_devWorkQueue);
	cudaFree(g_devCount);

	cudaFreeHost(g_pinnedPrimPtrs);
	cudaFree(g_devPrimPtrs);

	for(int i = 0; i < kMaxPrims; ++i)
	{
		if(g_devPrimSlabs[i])
		{
			cudaFree(g_devPrimSlabs[i]);
			g_devPrimSlabs[i] = nullptr;
		}
	}

	g_available = false;
}

// ---------------------------------------------------------------------------
// buildSingleQuadPrimitive
//
// Construct a Primitive that makes the existing Reactor rasteriser visit
// exactly one 2×2 quad at (x, y) with the coverage described by cMask.
//
// Coverage mapping (from QuadRasterizer::rasterize()):
//   bit 0 → pixel (x,   y):   left_y  ≤ x   < right_y
//   bit 1 → pixel (x+1, y):   left_y  ≤ x+1 < right_y
//   bit 2 → pixel (x,   y+1): left_y1 ≤ x   < right_y1
//   bit 3 → pixel (x+1, y+1): left_y1 ≤ x+1 < right_y1
//
// So for row y with bits {b0, b1}:
//   left_y  = b0 ? x : (b1 ? x+1 : x+2)
//   right_y = b1 ? x+2 : (b0 ? x+1 : x)
// ---------------------------------------------------------------------------

void CudaRasterizer::buildSingleQuadPrimitive(
    Primitive &dst,
    const Primitive &src,
    int x, int y,
    uint8_t cMask)
{
	// Copy all plane equations, stencil masks, reference points, etc.
	std::memcpy(&dst, &src, sizeof(Primitive));

	// Restrict vertical range to this single scanline pair.
	dst.yMin = y;
	dst.yMax = y + 2;

	// Zero the entire outline then write only the two scanlines we need.
	// The rasteriser reads outline[yMin] and outline[yMin+1].
	std::memset(dst.outline, 0, sizeof(dst.outline));

	auto encodeRow = [x](uint8_t b0, uint8_t b1) -> Primitive::Span {
		Primitive::Span s;
		s.left  = static_cast<unsigned short>(b0 ? x : (b1 ? x + 1 : x + 2));
		s.right = static_cast<unsigned short>(b1 ? x + 2 : (b0 ? x + 1 : x));
		return s;
	};

	dst.outline[y]     = encodeRow(cMask & 1, cMask & 2);
	dst.outline[y + 1] = encodeRow(cMask & 4, cMask & 8);
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
	// 1. Upload primitive outline spans to GPU
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

		// Size of GPUPrimitive with 'spanCount' entries, plus one overflow guard span.
		// spans[0] is already counted in sizeof(GPUPrimitive) (flexible array with [1]).
		// The +1 guard span is read by the kernel when (yMax - yMin) is odd: the last
		// scanline pair has spanIdx1 = spanCount, which would be one past the end without
		// it.  Initialising it to {0, 0} (empty span) prevents out-of-bounds garbage from
		// producing full-width work items that appear as horizontal streaks in the output.
		const size_t slab = sizeof(GPUPrimitive) + sizeof(GPUSpan) * spanCount;  // spanCount+1 slots total

		// Reallocate device slab if needed (grow only).
		if(g_devPrimSlabs[i])
		{
			// Re-use; we always free+reallocate on size changes for simplicity.
			// In a production path you'd track allocated sizes.
			cudaFree(g_devPrimSlabs[i]);
			g_devPrimSlabs[i] = nullptr;
		}
		CUDA_CHECK(cudaMalloc(&g_devPrimSlabs[i], slab));

		// Build a temporary host-side GPUPrimitive in a small stack buffer
		// (spanCount can be large; use heap for safety).
		std::vector<char> hostBuf(slab);
		GPUPrimitive *hp = reinterpret_cast<GPUPrimitive *>(hostBuf.data());
		hp->yMin = p.yMin;
		hp->yMax = p.yMax;
		for(int s = 0; s < spanCount; ++s)
		{
			hp->spans[s].left  = p.outline[p.yMin + s].left;
			hp->spans[s].right = p.outline[p.yMin + s].right;
		}
		// Overflow guard: kernel may read spans[spanCount] when spanCount is odd.
		hp->spans[spanCount] = { 0, 0 };

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
	// 5. Process each covered quad using the existing pixel routine
	// ------------------------------------------------------------------
	// We call pixelRoutine with a single-quad Primitive (count=1, cluster=0,
	// clusterCount=1).  The routine internally calls rasterize() which finds
	// exactly the one quad we encoded.
	Primitive singleQuad;

	for(int q = 0; q < clampedQuads; ++q)
	{
		const QuadWorkItem &item = g_pinnedWorkQueue[q];
		buildSingleQuadPrimitive(singleQuad, primitives[item.primIdx],
		                         item.x, item.y, item.cMask);
		pixelRoutine(device, &singleQuad, 1, 0, 1, data);
	}
}

}  // namespace sw
