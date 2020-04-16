#ifndef BASE_CUDA_FILTER_HPP_
#define BASE_CUDA_FILTER_HPP_

#pragma once

// DeepStream includes:

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include <cuda_runtime.h>

class BaseCudaFilter {
protected:
    // cuda stream to use for a filter
    cudaStream_t stream;
public:
    BaseCudaFilter();
    virtual ~BaseCudaFilter();

    /**
     * Called on every NVMM batched buffer.
     * 
     * Should be connected to a buffer callback or used in a filter plugin.
     * 
     * return a GstFlowReturn (success, failure, etc.)
     */
    virtual GstFlowReturn on_buffer(GstBuffer * buf);

    /**
     * Called on every frame by on_buffer
     * 
     * return true on success, false on failure
     */
    virtual bool on_frame(NvBufSurface * surf, NvDsFrameMeta * frame_meta);

    /**
     * Called on every object meta by on_frame
     * 
     * return true on success, false on failure
     */
    virtual bool on_object(NvDsFrameMeta * f_meta, NvDsObjectMeta * o_meta, NvBufSurfaceParams * frame) = 0;
};

#endif // BASE_CUDA_FILTER_HPP_