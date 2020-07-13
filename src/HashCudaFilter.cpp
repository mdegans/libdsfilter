/* HashCudaFilter.cpp
 *
 * This code borrows liberally from gstdsexample.cpp so...
 *
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2020, Michael de Gans <47511965+mdegans@users.noreply.github.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include "HashCudaFilter.hpp"

#include <new>

static bool
destroy_surface(NvBufSurface** surf_ref) {
  if (*surf_ref) {
    GST_DEBUG("destroying temporary surface");
    if (NvBufSurfaceDestroy(*surf_ref)) {
      GST_ERROR("failed to destroy surface");
      return false;
    }
    *surf_ref = nullptr;
  }
}

HashCudaFilter::HashCudaFilter() {
  GST_DEBUG(__func__);

  params = new NvBufSurfTransformParams();
  config = new NvBufSurfTransformConfigParams();
  create_params = new NvBufSurfaceCreateParams();
  tmp_surface = nullptr;

  // setup the session config
  config->compute_mode = NvBufSurfTransformCompute_Default;
  config->cuda_stream = this->stream;

  // common stuff on the frame config
  // box sampling would probably be best in this case, but it's not available.
  // performance of this hash will be poor until it is, or alternative code is
  // written so that every pixel in the source image contributes to the hash.
  params->transform_filter = NvBufSurfTransformInter_Bilinear;
  params->transform_flag = NVBUFSURF_TRANSFORM_FILTER;

  // buffer creation parameters
  // we want greyscale in the end
  create_params->colorFormat = NVBUF_COLOR_FORMAT_GRAY8;
  // 8*8 bytes ought to be enough
  create_params->width = HashCudaFilter::WIDTH;
  create_params->height = HashCudaFilter::HEIGHT;
  create_params->size = HashCudaFilter::SIZE;
#ifdef IS_TEGRA
  // this is surface memory on Jetson (textures)
  create_params->memType = NVBUF_MEM_DEFAULT;
#else
  // x86 use unified memory (easier memory copies)
  create_params->memType = NVBUF_MEM_CUDA_UNIFIED;
#endif
  // 1d array
  create_params->layout = NVBUF_LAYOUT_PITCH;
}

HashCudaFilter::~HashCudaFilter() {
  GST_DEBUG(__func__);
  delete[] params->dst_rect;
  delete[] params->src_rect;
  delete params;
  delete config;
  delete create_params;
  destroy_surface(&tmp_surface);
}

bool
HashCudaFilter::create_tmp_surface(const NvBufSurface* surf) {
  // reset the tmp surface if any
  if (!destroy_surface(&(this->tmp_surface))) {
    return false;
  }

  // set the gpuId from upstream
  create_params->gpuId = surf->gpuId;
  config->gpu_id = surf->gpuId;

  GST_DEBUG("creating new surface");
  if (NvBufSurfaceCreate(
      &(this->tmp_surface), surf->batchSize, this->create_params)) {
    GST_ERROR("failed to create surface");
    return false;
  }

  return true;
}

bool
HashCudaFilter::set_tx_rects(NvBufSurface* surf) {
  // the destination rectangle is always the same
  static const NvBufSurfTransformRect dest_rect = {
    0, 0, HashCudaFilter::WIDTH, HashCudaFilter::HEIGHT};

  if (!this->params->src_rect) {
    try {
      this->params->src_rect = new NvBufSurfTransformRect[surf->batchSize]();
    } catch(const std::bad_alloc& e) {
      GST_ERROR("%s", e.what());
      return false;
    }
  }

  for (size_t i = 0; i < surf->numFilled; i++)
  {
    this->params->dst_rect[i] = dest_rect;
    this->params->src_rect[i].height = surf->surfaceList[i].height;
    this->params->src_rect[i].width = surf->surfaceList[i].width;
  }
  return true;
}

bool
HashCudaFilter::on_batch(NvBufSurface* surf, NvDsBatchMeta* batch_meta) {
  static NvBufSurfTransform_Error err;
  // So, to avoid needing to set the gpu_id and batch size explicitly on the
  // element, we get it from the first upstream buffer. I'm sure there is a
  // downside to this, but I can't see it (yet).
  if (this->tmp_surface == nullptr && !this->create_tmp_surface(surf)) {
    return false;
  }

  // set the transform parameters from the input surface
  if (!set_tx_rects(surf)) {
    return false;
  }

  // transform the surface.
  err = NvBufSurfTransform(surf, this->tmp_surface, this->params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ERROR("Got error code %d from NvBufSurfTransform", err);
    return false;
  }
}

GstFlowReturn
HashCudaFilter::on_buffer(GstBuffer* buf) {
  GST_LOG("on_buffer:got buffer");

  GstMapInfo info;
  NvBufSurface* surf = nullptr;
  NvDsBatchMeta* batch_meta = nullptr;
  NvDsFrameMeta* frame_meta = nullptr;
  NvDsFrameMetaList* l_frame = nullptr;

  // get the info about the buffer
  memset(&info, 0, sizeof(info));
  if (!gst_buffer_map(buf, &info, GST_MAP_READ)) {
    GST_ERROR("on_buffer:failed to get info from buffer");
    return GST_FLOW_ERROR;
  }

  // get the surface data
  surf = (NvBufSurface*)info.data;

  // get the batch metadata from the buffer
  batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  // lock the meta
  nvds_acquire_meta_lock(batch_meta);

  GST_LOG("on_buffer:got batch with %d frames.",
          batch_meta->num_frames_in_batch);

  if (!on_batch(surf, batch_meta)) {
    nvds_release_meta_lock(batch_meta);
    return GST_FLOW_ERROR;
  }
  nvds_release_meta_lock(batch_meta);
  return GST_FLOW_OK;
}