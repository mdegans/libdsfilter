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
  GST_LOG("on_buffer:got batch with %d frames.",
          batch_meta->num_frames_in_batch);

  // for frame_meta in batch_meta
  for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr;
       l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta*)l_frame->data;
    if (frame_meta == nullptr) {
      GST_ERROR("on_buffer:frame_meta is NULL");
      return GST_FLOW_ERROR;
    }

    if (!on_frame(surf, frame_meta)) {
      return GST_FLOW_ERROR;
    };
  }

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
