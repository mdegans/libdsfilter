#include "queue.hpp"

#include "gtest/gtest.h"

#include <thread>

namespace my {
namespace project {
namespace {

// The fixture for testing class Queue.
class QueueTest : public ::testing::Test {
public:

  my::Queue<std::unique_ptr<int>> queue_;

  void fill_queue() {
    for (size_t i = 0; i < 10; i++)
    {
      std::unique_ptr<int> ptr(new int(i));
      queue_.put(std::move(ptr));
    }
  }
  /**
   * Consumes first 10 members from the queue, check expected item.
   */
  void for_consumer() {
    for (size_t i = 0; i < 10; i++) {
      auto elem = std::move(queue_.get());
      ASSERT_EQ(i, *elem);
    }
  }
  /**
   * Consumes members from the queue while(member).
   */
  void while_consumer() {
    auto elem = std::move(queue_.get());
    int i = 0;
    while (elem) {
      ASSERT_EQ(i, *elem);
      i++;
      elem = std::move(queue_.get());
    }
  }
};

// Test that the queue works in a single threaded environment.
TEST_F(QueueTest, SingleThreadPutFirst) {
  fill_queue();
  for_consumer();
}

// Test the Queue works in a multi-threaded environment by putting first.
TEST_F(QueueTest, MultiThreadPutFirst) {
  fill_queue();
  std::thread worker(&QueueTest::for_consumer, this);
  worker.join();
  ASSERT_TRUE(queue_.empty());
}

// Test the Queue works in a multi-threaded environment by getting first.
TEST_F(QueueTest, MultiThreadGetFirst) {
  std::thread worker(&QueueTest::for_consumer, this);
  fill_queue();
  worker.join();
  ASSERT_TRUE(queue_.empty());
}

// test the queue flush function flushes (this test needs improvement)
TEST_F(QueueTest, FlushTest) {
  fill_queue();
  queue_.flush();
  std::thread worker(&QueueTest::while_consumer, this);
  worker.join();
  ASSERT_TRUE(queue_.empty());
}

// test a while loop and the nullptr poison pill
TEST_F(QueueTest, PoisonPillTest) {
  fill_queue();
  std::thread worker(&QueueTest::while_consumer, this);
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