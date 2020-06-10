# https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_tools")

INCLUDE(AddGoogleTest)

ADD_EXECUTABLE(test_queue test_queue.cpp)
TARGET_INCLUDE_DIRECTORIES(test_queue PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/googletest-src/googletest/include/>
)
ADD_GTEST(test_queue)

