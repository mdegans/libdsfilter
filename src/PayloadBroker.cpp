/* PayloadBroker.cpp
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

// TODO(mdegans): think about merging ProtoPayloadFilter's conversion logic
//  this.

#include "PayloadBroker.hpp"
#include "nvdsmeta.h"


GstFlowReturn
PayloadBroker::on_buffer(GstBuffer* buf)
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
    // please note that this won't currently work with the default
    // message converter element without modification.
    // if the attached metadata is already serialized
    if (user_meta->base_meta.meta_type == NVDS_PAYLOAD_META) {
      // call on_batch_payload with our string
      auto payload = (std::string*) user_meta->user_meta_data;
      if (payload == nullptr) {
        GST_WARNING("payload was NULL");
        continue;
      }
      if (!this->on_batch_payload(payload)) {
        continue;
      }
    }
  }
  nvds_release_meta_lock(batch_meta);
  return GST_FLOW_OK;
}
