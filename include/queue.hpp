/* queue.hpp
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

#ifndef QUEUE_HPP_
#define QUEUE_HPP_

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>

namespace my {

/**
 * A simple thread-safe queue.
 */
template <typename T>
class Queue {
protected:
  std::deque<T> d;
  std::mutex mutex;
  std::condition_variable cv;
public:
  /**
   * Checks if the Queue is empty.
   */
  bool empty() {
    std::unique_lock<std::mutex> lock(this->mutex);
    return this->d.empty();
  }
  /**
   * Move an item into the queue. Does not block (no max size).
   */
  void put(T&& thing) {
    this->d.push_back(std::move(thing));
    this->cv.notify_one();
  }
  /**
   * Get an item from the Queue. Blocks while the queue is empty.
   */
  T get() {
    std::unique_lock<std::mutex> lock(this->mutex);
    while (this->d.empty()) this->cv.wait(lock);
    T ret = std::move(this->d.front());
    this->d.pop_front();
    return ret;
  }
};

} // namespace my

#endif