/* DistanceFilter.hpp
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

#ifndef DISTANCE_FILTER_HPP_
#define DISTANCE_FILTER_HPP_

#pragma once

#include "BaseFilter.hpp"

/**
 * DistanceFilter modifies osd metadata to make closer objects red.
 */
class DistanceFilter : public BaseFilter {
 public:
  DistanceFilter();
  virtual ~DistanceFilter() = default;
  /**
   * The class id to turn red.
   */
  int class_id;
  /**
   * Whether to set osd metadata for drawing.
   */
  bool do_drawing;
  /**
   * This implementation does drawing and analytics on NvDs Metadata.
   */
  GstFlowReturn on_buffer(GstBuffer* buf);
};

#endif  // DISTANCE_FILTER_HPP_