#ifndef PROTO_PAYLOAD_FILTER_HPP_
#define PROTO_PAYLOAD_FILTER_HPP_

#pragma once

#include "BaseFilter.hpp"
#include "distance.pb.h"

/**
 * ProtoPayloadFilter serializes distanceproto metadata.
 */
class ProtoPayloadFilter : public BaseFilter {
public:
  ProtoPayloadFilter();
  virtual ~ProtoPayloadFilter() = default;
  /**
   * This implementation extracts metadata of type DF_USER_BATCH_META
   * from the user metadata list on batch_meta and calls on_batch with each.
   */
  virtual GstFlowReturn on_buffer(GstBuffer* buf);
  /**
   * Called by on_buffer when payload metadata is found in batch_meta's user
   * meta list.
   * 
   * The default implementation serializes the Batch to string and attaches
   * the result as NvDsPayload type metadata (for message broker elements).
   * 
   * Returns true on success, false on failure.
   */
  virtual bool on_batch_meta(
    NvDsBatchMeta* batch_meta, distanceproto::Batch* batch);
};

#endif  // PROTO_PAYLOAD_FILTER_HPP_