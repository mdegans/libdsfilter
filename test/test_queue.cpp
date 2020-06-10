#include "queue.hpp"

#include "gtest/gtest.h"

#include <thread>

namespace my {
namespace project {
namespace {

// The fixture for testing class Queue.
class QueueTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if their bodies would
  // be empty.

  /**
   * Items to put in the queue
   */
  std::vector<int> items_;
  /**
   * The actual queue
   */
  my::Queue<int> queue_;

  void SetUp() override {
    for (size_t i = 0; i < 10; i++)
    {
      items_.push_back(i);
    }
  }
 public:
  /**
   * Consumes members from the queue, check expected item.
   */
  void consumer() {
    for (auto i : items_)
    {
      ASSERT_EQ(i, std::move(queue_.get()));
    }
  }
};

// Test that the queue works in a single threaded environment.
TEST_F(QueueTest, SingleThreadPutFirst) {
  for (auto i : items_)
  {
    queue_.put(std::move(i));
  }
  for (auto i : items_)
  {
    ASSERT_EQ(i, std::move(queue_.get()));
  }
}

// Test the Queue works in a multi-threaded environment by putting first.
TEST_F(QueueTest, MultiThreadPutFirst) {
  for (auto i : items_) {
    queue_.put(std::move(i));
  }
  std::thread worker(&QueueTest::consumer, this);
  worker.join();
  ASSERT_TRUE(queue_.empty());
}

// Test the Queue works in a multi-threaded environment by getting first.
TEST_F(QueueTest, MultiThreadGetFirst) {
  std::thread worker(&QueueTest::consumer, this);
  for (auto i : items_) {
    queue_.put(std::move(i));
  }
  worker.join();
  ASSERT_TRUE(queue_.empty());
}

}  // namespace
}  // namespace project
}  // namespace my

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}