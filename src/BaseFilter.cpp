/* BaseFilter.cpp
 *
 * Copyright 2020 Michael de Gans <47511965+mdegans@users.noreply.github.com>
 *
 * 66E67F6ADF56899B2AA37EF8BF1F2B9DFBB1D82E66BD48C05D8A73074A7D2B75
 * EB8AA44E3ACF111885E4F84D27DC01BB3BD8B322A9E8D7287AD20A6F6CD5CB1F
 *
 * This file is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 3 of the
 * License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "BaseFilter.hpp"

GstFlowReturn BaseFilter::on_buffer(GstBuffer* buf) {
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
  // we need to lock the metadata
  nvds_acquire_meta_lock(batch_meta);

  GST_LOG("on_buffer:got batch with %d frames.",
          batch_meta->num_frames_in_batch);

  // for frame_meta in batch_meta
  for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr;
       l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta*)l_frame->data;
    if (frame_meta == nullptr) {
      GST_ERROR("on_buffer:frame_meta is NULL");
      // release lock before return
      nvds_release_meta_lock(batch_meta);
      return GST_FLOW_ERROR;
    }

    if (!on_frame(surf, frame_meta)) {
      // release lock before return
      nvds_release_meta_lock(batch_meta);
      return GST_FLOW_ERROR;
    };
  }
  // release lock before return
  nvds_release_meta_lock(batch_meta);
  return GST_FLOW_OK;
}

bool BaseFilter::on_frame(NvBufSurface* surf, NvDsFrameMeta* frame_meta) {
#ifdef VERBOSE
  GST_LOG("on_frame:processing frame %d", frame_meta->frame_num);
#endif

  // if there are no detected objects in the frame, skip it
  if (!frame_meta->num_obj_meta) {
    return true;
  }

  NvBufSurfaceParams* frame = &surf->surfaceList[frame_meta->batch_id];
  NvDsObjectMeta* obj_meta = nullptr;
  NvDsObjectMetaList* l_obj = nullptr;

  // for obj_meta in frame meta
  for (l_obj = frame_meta->obj_meta_list; l_obj != nullptr; l_obj = l_obj->next) {
    obj_meta = (NvDsObjectMeta*)l_obj->data;
    if (obj_meta == nullptr) {
      GST_ERROR("on_frame:obj_meta is NULL");
      return false;
    }

    if (!on_object(frame_meta, obj_meta, frame)) {
      return false;
    }
  }
  return true;
}

bool BaseFilter::on_object(NvDsFrameMeta* f_meta,
                           NvDsObjectMeta* o_meta,
                           NvBufSurfaceParams* frame) {
  GST_LOG("BaseFilter::on_object:got frame %d", f_meta->frame_num);
  GST_LOG("BaseFilter::on_object:surface color format: %d",
          frame->colorFormat);
  GST_LOG("BaseFilter::on_object:got object: %s with confidence %.2f",
          o_meta->obj_label, o_meta->confidence);
  return true;
}
