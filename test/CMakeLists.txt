# https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_tools")

# This is so much easier than the trashfire that is GoogleTest included with
# Cmake.
INCLUDE(AddGoogleTest)

# test the queue
ADD_EXECUTABLE(test_queue test_queue.cpp)
TARGET_INCLUDE_DIRECTORIES(test_queue PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/googletest-src/googletest/include/>
)
ADD_GTEST(test_queue)

# test the file meta broker
ADD_EXECUTABLE(test_filemetabroker test_FileMetaBroker.cpp)
TARGET_LINK_LIBRARIES(test_filemetabroker PRIVATE
  ${PROJECT_NAME} ${GSTREAMER_LIBRARIES} ${DISTANCEPROTO_LIBRARIES}
  stdc++fs
  nvds_meta
  nvdsgst_meta
  nvbufsurface
  nvbufsurftransform
)
TARGET_INCLUDE_DIRECTORIES(test_filemetabroker PRIVATE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/googletest-src/googletest/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  /opt/nvidia/deepstream/deepstream/sources/includes
  ${GSTREAMER_INCLUDE_DIRS}
  ${DISTANCEPROTO_INCLUDE_DIRS}
)
ADD_GTEST(test_filemetabroker)