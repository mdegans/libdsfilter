/* DistanceFilter.cpp
 *
 * Copyright 2020 Michael de Gans
 *
 * 4019dc5f7144321927bab2a4a3a3860a442bc239885797174c4da291d1479784
 * 5a4a83a5f111f5dbd37187008ad889002bce85c8be381491f8157ba337d9cde7
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE X CONSORTIUM BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name(s) of the above copyright
 * holders shall not be used in advertising or otherwise to promote the sale,
 * use or other dealings in this Software without prior written
 * authorization.
 */

/**
 * TODOS:
 *  - use gstreamer object logging instead of GST_WARNING
 */

#include "DistanceFilter.hpp"
#include "distance.pb.h"

#include <math.h>

namespace dp = distanceproto;

/**
 * Calculate how dangerous a list element is based on proximity to other
 * list elements.
 */
static float
calculate_how_dangerous(int class_id, NvDsMetaList* l_obj, float danger_distance);

/**
 *  distanceproto batch nvds user metadata type
 */
#define NVDS_USER_BATCH_META_DP (nvds_get_user_meta_type((gchar*)"NVIDIA.NVINFER.USER_META"))

/**
 * NvDsUserMeta copy function for batch level distance metadata.
 */
static gpointer copy_dp_batch_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;

  auto src_b_proto = (dp::Batch*)(user_meta->user_meta_data);
  auto dst_b_proto = new dp::Batch();

  dst_b_proto->CopyFrom(*src_b_proto);

  return (gpointer) dst_b_proto;
}

/**
 * NvDsUserMeta release function for batch level distance metadata.
 */
static void release_dp_batch_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;

  auto b_proto = (dp::Batch*)(user_meta->user_meta_data);
  delete b_proto;
}

DistanceFilter::DistanceFilter() {
  // copypasta from the protobuf docs:
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // set some default values for this
  this->do_drawing = true;
  this->class_id = 0;
}

GstFlowReturn
DistanceFilter::on_buffer(GstBuffer* buf)
{
  float how_dangerous=0.0f;
  float color_val=0.0f;

  // GList of NvDsFrameMeta
  NvDsMetaList* l_frame = nullptr;
  // GList of NvDsObjectMeta
  NvDsMetaList* l_obj = nullptr;
  // Nvidia batch level metadata
  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == nullptr) {
    GST_WARNING("dsdistance: no metadata attached to buffer !!!");
    return GST_FLOW_OK;
  }
  // we need to lock the metadata
  GRecMutex* meta_lock = &batch_meta->meta_mutex;
  g_rec_mutex_lock(meta_lock);
  // Nvidia user metadata structure
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == nullptr) {
    GST_WARNING("dsdistance: could not get user meta from batch pool !!!");
    g_rec_mutex_unlock(meta_lock);
    return GST_FLOW_OK;
  }
  // Nvidia frame level metadata
  NvDsFrameMeta *frame_meta = nullptr;
  // Nvidia object level metadata
  NvDsObjectMeta* obj_meta = nullptr;
  // Nvidia BBox structure (for osd element)
  NvOSD_RectParams* rect_params = nullptr;
  // NvOSD_TextParams* text_params;

  // our Batch level metadata
  auto b_proto = new dp::Batch();
  // attach it to nvidia user meta
  user_meta->user_meta_data = (void*) b_proto;
  user_meta->base_meta.meta_type = NVDS_USER_BATCH_META_DP;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) copy_dp_batch_meta;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) release_dp_batch_meta;
  // add nvidia user meta to the batch
  nvds_add_user_meta_to_batch(batch_meta, user_meta);

  // for frame_meta in frame_meta_list
  for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr;
      l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);

    // if somehow the frame meta doesn't exist, warn and continue
    // (we could also skip the whole batch)
    if (frame_meta == nullptr) {
      GST_WARNING("NvDS Meta contained NULL meta");
      continue;
    }

    // our Frame level metadata
    auto f_proto = b_proto->add_frames();

    // danger score for this frame
    float f_danger = 0.0f;

    // for obj_meta in obj_meta_list
    for (l_obj = frame_meta->obj_meta_list; l_obj != nullptr;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      // skip the object, if it's not a person
      if (obj_meta->class_id != this->class_id) {
        continue;
      }

      // our Person level metadata
      dp::Person* p_proto = f_proto->add_people();
      // metadata for the person's bounding box
      auto b_proto = new dp::BBox();

      rect_params = &(obj_meta->rect_params);
      // text_params = &(obj_meta->text_params); // TODO(mdegans, osd labels?)
      // record the bounding box and set it on the person
      b_proto->set_height(rect_params->height);
      b_proto->set_left(rect_params->left);
      b_proto->set_top(rect_params->top);
      b_proto->set_width(rect_params->width);
      p_proto->set_allocated_bbox(b_proto);

      // get how dangerous the object is as a float
      how_dangerous = calculate_how_dangerous(
          this->class_id, l_obj, rect_params->height);
      // set it on the person metadata
      p_proto->set_danger_val(how_dangerous);
      // add it to the frame danger score
      f_danger += how_dangerous;

      if (this->do_drawing) {
        // make the box opaque and red depending on the danger
        color_val = (how_dangerous * 0.6f);
        color_val = color_val < 0.6f ? color_val : 0.6f;

        rect_params->border_width = 0;
        rect_params->has_bg_color = 1;
        rect_params->bg_color.red = color_val + 0.2f;
        rect_params->bg_color.green = 0.2f;
        rect_params->bg_color.blue = 0.2f;
        rect_params->bg_color.alpha = color_val + 0.2f;
      }
    }
    // set the sum danger for the frame
    f_proto->set_sum_danger(f_danger);
  }
  g_rec_mutex_unlock(meta_lock);
  return GST_FLOW_OK;
}

/**
 * Calculate distance between the center of the bottom edge of two rectangles
 */
static float
distance_between(NvOSD_RectParams* a, NvOSD_RectParams* b) {
  // use the middle of the feet as a center point.
  int ax = a->left + a->width / 2;
  int ay = a->top + a->height;
  int bx = b->left + b->width / 2;
  int by = b->top + b->height;

  int dx = ax - bx;
  int dy = ay - by;

  return sqrtf((float)(dx * dx + dy * dy));
}

static float
calculate_how_dangerous(int class_id, NvDsMetaList* l_obj, float danger_distance) {
  NvDsObjectMeta* current = (NvDsObjectMeta *) (l_obj->data);
  NvDsObjectMeta* other;

  // sum of all normalized violation distances
  float how_dangerous = 0.0f;

  float d; // distance temp (in pixels)

  // iterate forwards from current element
  for (NvDsMetaList* f_iter = l_obj->next; f_iter != nullptr; f_iter = f_iter->next) {
    other = (NvDsObjectMeta *) (f_iter->data);
    if (other->class_id != class_id) {
        continue;
    }
    d = danger_distance - distance_between(&(current->rect_params), &(other->rect_params));
    if (d > 0.0) {
      how_dangerous += d / danger_distance;
    }
  }

  // iterate in reverse from current element
  for (NvDsMetaList* r_iter = l_obj->prev; r_iter != nullptr; r_iter = r_iter->prev) {
    other = (NvDsObjectMeta *) (r_iter->data);
    if (other->class_id != class_id) {
        continue;
    }
    d = danger_distance - distance_between(&(current->rect_params), &(other->rect_params));
    if (d > 0.0f) {
      how_dangerous += d / danger_distance;
    }
  }

  return how_dangerous;
}
