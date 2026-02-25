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

#ifndef sw_CudaRasterizer_hpp
#define sw_CudaRasterizer_hpp

#include "Primitive.hpp"
#include "Renderer.hpp"

namespace vk {
class Device;
}

namespace sw {

// CudaRasterizer offloads triangle traversal to the GPU.
//
// The traversal kernel (CudaTraversal.cu) reads outline spans from each
// Primitive and emits a flat list of covered 2×2 pixel quads (QuadWorkItem).
// The CPU then processes each quad by calling the existing JIT pixel routine
// with a single-quad Primitive, preserving all shader / depth / stencil logic.
//
// Only the default-stream, single-sample path is implemented.  Draws with
// multisampling fall back to the standard CPU cluster loop.
class CudaRasterizer
{
public:
	// Returns true if at least one CUDA-capable device is present.
	// Result is cached after the first call.
	static bool isAvailable();

	// One-time initialisation: allocate persistent device/pinned buffers and
	// create the CUDA stream.  Called lazily on first use.
	static void init();

	// Release all GPU resources.  Call at renderer shutdown.
	static void shutdown();

	// Replace the 16-cluster CPU pixel loop for one batch.
	//
	// primitives  – array of (count) Primitive structs from processPrimitives()
	// count       – number of visible primitives in the batch
	// pixelRoutine – JIT-compiled function that runs depth/stencil/shader/blend
	// data        – DrawData for this draw call
	// device      – Vulkan device (passed through to pixelRoutine)
	static void processPixels(
	    const Primitive *primitives,
	    int count,
	    PixelProcessor::RoutineType pixelRoutine,
	    DrawData *data,
	    vk::Device *device);

};

}  // namespace sw

#endif  // sw_CudaRasterizer_hpp
