/* DistanceFilter.cpp
 *
 * Copyright 2020 Michael de Gans
 * 
 * many thanks to:
 * deepstream_user_metadata_app.c
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
 *  - handle existing metadata, since that isn't covered yet
 */

#include "DistanceFilter.hpp"
#include "distance.pb.h"

#include <math.h>

namespace dp = distanceproto;

static const bool DEFAULT_DO_DRAWING=false;
static const float DEFAULT_FILTER_HEIGHT_DIFF=0.25f;
static const float DEFAULT_CLASS_ID=0;

/**
 * Calculate how dangerous a list element is based on proximity to other
 * list elements.
 */
static float
calculate_how_dangerous(int class_id, NvDsMetaList* l_obj, float danger_distance);

/**
 * NvDsUserMeta copy function for batch level distance metadata.
 */
static gpointer copy_dp_batch_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta* user_meta = (NvDsUserMeta *)data;

  auto src_batch_proto = (dp::Batch*)(user_meta->user_meta_data);
  auto dst_batch_proto = new dp::Batch();

  dst_batch_proto->CopyFrom(*src_batch_proto);

  return (gpointer) dst_batch_proto;
}

/**
 * NvDsUserMeta release function for batch level distance metadata.
 */
static void release_dp_batch_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta* user_meta = (NvDsUserMeta *)data;

  auto batch_proto = (dp::Batch*)(user_meta->user_meta_data);
  delete batch_proto;
}

DistanceFilter::DistanceFilter() {
  // copypasta from the protobuf docs:
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  // set some default values for this
  this->do_drawing = DEFAULT_DO_DRAWING;
  this->class_id = DEFAULT_CLASS_ID;
  this->filter_height_diff = DEFAULT_FILTER_HEIGHT_DIFF;
}

// TODO(mdegans): split this function up and clean it up
GstFlowReturn
DistanceFilter::on_buffer(GstBuffer* buf)
{
  /**
   * https://gstreamer.freedesktop.org/documentation/gstreamer/gstbuffer.html
   *
   * If a plug-in wants to modify the buffer data or metadata in-place,
   * it should first obtain a buffer that is safe to modify by using
   * gst_buffer_make_writable.
   */
  gst_buffer_make_writable(buf);

  float person_danger=0.0f;
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
  nvds_acquire_meta_lock(batch_meta);
  // Nvidia user metadata structure
  NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == nullptr) {
    GST_WARNING("dsdistance: could not get user meta from batch pool !!!");
    nvds_release_meta_lock(batch_meta);
    return GST_FLOW_OK;
  }
  // Nvidia frame level metadata
  NvDsFrameMeta* frame_meta = nullptr;
  // Nvidia object level metadata
  NvDsObjectMeta* obj_meta = nullptr;
  // Nvidia BBox structure (for osd element)
  NvOSD_RectParams* rect_params = nullptr;
  NvOSD_TextParams* text_params = nullptr;

  // our Batch level metadata
  auto batch_proto = new dp::Batch();
  // attach it to nvidia user meta
  user_meta->user_meta_data = (void*) batch_proto;
  user_meta->base_meta.meta_type = DF_USER_BATCH_META;
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
    auto frame_proto = batch_proto->add_frames();

    // danger score for this frame
    float frame_danger = 0.0f;

    // for obj_meta in obj_meta_list
    for (l_obj = frame_meta->obj_meta_list; l_obj != nullptr;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      // skip the object, if it's not a person
      if (obj_meta->class_id != this->class_id) {
        continue;
      }

      // our Person level metadata
      dp::Person* person_proto = frame_proto->add_people();
      // metadata for the person's bounding box
      auto bb_proto = new dp::BBox();

      rect_params = &(obj_meta->rect_params);
      // text_params = &(obj_meta->text_params); // TODO(mdegans, osd labels?)
      // record the bounding box and set it on the person
      bb_proto->set_height(rect_params->height);
      bb_proto->set_left(rect_params->left);
      bb_proto->set_top(rect_params->top);
      bb_proto->set_width(rect_params->width);
      person_proto->set_allocated_bbox(bb_proto);

      // get how dangerous the object is as a float
      person_danger = calculate_how_dangerous(
          this->class_id, l_obj, this->filter_height_diff);
      // set it on the person metadata
      person_proto->set_danger_val(person_danger);
      // TODO(mdegans): make this configurable
      if (person_danger >= 1.0) {
        person_proto->set_is_danger(true);
      }
      // add it to the frame danger score
      frame_danger += person_danger;

      if (this->do_drawing) {
        // make the box opaque and red depending on the danger
        color_val = (person_danger * 0.6f);
        color_val = color_val < 0.6f ? color_val : 0.6f;

        // gchararray display_text = nullptr;
        // sprintf(display_text, "%.2f", person_danger);
        // text_params->display_text = display_text;

        rect_params->border_width = 0;
        rect_params->has_bg_color = 1;
        rect_params->bg_color.red = (double) color_val + 0.2;
        rect_params->bg_color.green = 0.2;
        rect_params->bg_color.blue = 0.2;
        rect_params->bg_color.alpha = (double) color_val + 0.2;
      }
    }
    // set the sum danger for the frame
    frame_proto->set_sum_danger(frame_danger);
  }
  nvds_release_meta_lock(batch_meta);
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

/**
 * Return true if how_much % height difference.
 *
 * @param how_much is the height % difference between two bounding boxes
 * after which a true value is returned.
 *
 * if (abs(current->rect_params.height - other->rect_params.height) > current->rect_params.height * how_much) {
 *   return true;
 * }
 * 
 * A true return value may be used to skip a detection. The idea is that
 * Higher height difference either meas kid or a narrow camera angle or far away.
 * camera angles where people in the foreground against a background like a crowd
 * are missed, so this can help to fix that by filtering out these pairs.
 *
 */
static bool
too_far(const NvDsObjectMeta* current, const NvDsObjectMeta* other, const float how_much) {
  if (abs(current->rect_params.height - other->rect_params.height) > current->rect_params.height * how_much) {
    return true;
  }
  return false;
}

static float
calculate_how_dangerous(int class_id, NvDsMetaList* l_obj, float background_cutoff=0.25f) {
  NvDsObjectMeta* current = (NvDsObjectMeta *) (l_obj->data);
  NvDsObjectMeta* other;

  float how_dangerous = 0.0f;  // sum of all normalized violation distances.
  float danger_distance = current->rect_params.height;
  float d; // distance temp (in pixels).

  // TODO(mdegans): combine these two loops into one function?
  // there may be a function to do this (iteration in both directions from a starting
  // element of a GList.)

  // iterate forwards from current element
  for (NvDsMetaList* f_iter = l_obj->next; f_iter != nullptr; f_iter = f_iter->next) {
    other = (NvDsObjectMeta *) (f_iter->data);
    if (other->class_id != class_id) {
        continue;
    }
    if (too_far(current, other, background_cutoff)) {
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
    if (too_far(current, other, background_cutoff)) {
      continue;
    }
    d = danger_distance - distance_between(&(current->rect_params), &(other->rect_params));
    if (d > 0.0f) {
      how_dangerous += d / danger_distance;
    }
  }

  return how_dangerous;
}
