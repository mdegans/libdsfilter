/* ProtoPayloadFilter.cpp
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

#include "ProtoPayloadFilter.hpp"
#include "DistanceFilter.hpp"  // NVDS_USER_BATCH_META_DP

#include "distance.pb.h"
#include "nvdsmeta.h"


namespace dp = distanceproto;

/**
 * NvDsUserMeta copy function for string payload metadata.
 */
static gpointer copy_dp_payload_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta* user_meta = (NvDsUserMeta *)data;

  auto src_payload = (std::string*)(user_meta->user_meta_data);
  auto dst_payload = new std::string();

  *dst_payload = *src_payload;

  return (gpointer) dst_payload;
}

/**
 * NvDsUserMeta release function for string payload metadata.
 */
static void release_dp_payload_meta(gpointer data, gpointer user_data) {
  NvDsUserMeta* user_meta = (NvDsUserMeta *)data;

  auto payload = (std::string*)(user_meta->user_meta_data);
  delete payload;
}

ProtoPayloadFilter::ProtoPayloadFilter() {
  // copypasta from the protobuf docs:
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

GstFlowReturn
ProtoPayloadFilter::on_buffer(GstBuffer* buf)
{
  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  if (batch_meta == nullptr) {
    GST_WARNING("protopayload: no metadata attached to buffer !!!");
    return GST_FLOW_OK;
  }

  // we need to lock the metadata
  nvds_acquire_meta_lock(batch_meta);

  // loop through the batch level user metadata list
  for (auto elem = batch_meta->batch_user_meta_list; elem != nullptr; elem = elem->next)
  {
    // get user meta from the list
    auto user_meta = (NvDsUserMeta*) elem->data;
    if (user_meta == nullptr){
      GST_WARNING("empty meta found in GList");
      continue;
    }
    // if the attached metadata is not ours, skip it
    if (user_meta->base_meta.meta_type != DF_USER_BATCH_META) {
      continue;
    }
    auto batch = (dp::Batch*) user_meta->user_meta_data;

    if (!this->on_batch_meta(batch_meta, batch)) {
      continue;
    }
  }

  nvds_release_meta_lock(batch_meta);
  return GST_FLOW_OK;
}

bool
ProtoPayloadFilter::on_batch_meta(NvDsBatchMeta* batch_meta, dp::Batch* batch) {
  // try to get a new user metadata pointer from the pool
  auto user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (user_meta == nullptr) {
    GST_WARNING("could not get user metadata from batch pool");
    return false;
  }

  // try to serialize the batch to string
  auto payload = new std::string();
  if (!batch->SerializeToString(payload)) {
    GST_WARNING("could not convert payload to string");
    delete payload;
    return false;
  }

  // attach the payload to the user_meta as type NVDS_PAYLOAD_META
  user_meta->user_meta_data = (void*) payload;
  // not sure if this will work, but we'll find out if the broker accepts it
  // I'm inferring this from deepstream_user_metadata_app.c, intellisense
  // and fgrep for "payload"
  user_meta->base_meta.meta_type = NVDS_PAYLOAD_META;
  user_meta->base_meta.copy_func = copy_dp_payload_meta;
  user_meta->base_meta.release_func = release_dp_payload_meta;
  user_meta->base_meta.uContext = this;

  // attach the payload to the buffer at the batch level
  nvds_add_user_meta_to_batch(batch_meta, user_meta);

  return true;
}