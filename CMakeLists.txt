PROJECT(dsfilter CXX CUDA)

# get version from file
FILE(READ "VERSION" DSFILTER_VERSION_STRING)

# https://stackoverflow.com/questions/18658233/split-string-to-3-variables-in-cmake
STRING(REPLACE "." ";" VERSION_LIST ${DSFILTER_VERSION_STRING})
LIST(GET VERSION_LIST 0 DSFILTER_VERSION_MAJOR)
LIST(GET VERSION_LIST 1 DSFILTER_VERSION_MINOR)
LIST(GET VERSION_LIST 2 DSFILTER_VERSION_PATCH)

MESSAGE("${PROJECT_NAME} version: ${DSFILTER_VERSION_MAJOR}.${DSFILTER_VERSION_MINOR}.${DSFILTER_VERSION_PATCH}")

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 17)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wshadow -Werror -fno-exceptions")

FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GSTREAMER REQUIRED gstreamer-1.0)
PKG_CHECK_MODULES(DISTANCEPROTO REQUIRED distanceproto-${DSFILTER_VERSION_MAJOR}.${DSFILTER_VERSION_MINOR})

# deepstream stuff
LINK_DIRECTORIES(/opt/nvidia/deepstream/deepstream/lib/)
# TARGET_LINK_DIRECTORIES doesn't seem to be supported in cmake 3.10\
# if anybody knows how to add this to the target so it can be exported
# it would be appreciated.

ADD_LIBRARY(${PROJECT_NAME} SHARED
  src/BaseCudaFilter.cpp
  src/BaseFilter.cpp
  src/DistanceFilter.cpp
  src/FileMetaBroker.cpp
  src/HashCudaFilter.cpp
  src/PayloadBroker.cpp
  src/ProtoPayloadFilter.cpp
  src/PyPayloadBroker.cpp
  src/TestCudaFilter.cu
)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES VERSION ${DSFILTER_VERSION_STRING} SOVERSION ${DSFILTER_VERSION_MAJOR})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_TARGET_PROPERTIES(
  ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER 
  "include/BaseFilter.hpp;include/DistanceFilter.hpp;include/BaseCudaFilter.hpp;include/TestCudaFilter.hpp;include/PayloadBroker.hpp;include/ProtoPayloadFilter.hpp;include/PyPayloadBroker.hpp;include/FileMetaBroker.hpp;include/queue.hpp"
)

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME};>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  /opt/nvidia/deepstream/deepstream/sources/includes
  ${GSTREAMER_INCLUDE_DIRS}
  ${DISTANCEPROTO_INCLUDE_DIRS}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${GSTREAMER_LIBRARIES} ${DISTANCEPROTO_LIBRARIES}
  nvds_meta
  nvdsgst_meta
  nvbufsurface
  nvbufsurftransform
)

# https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
OPTION(DSFILTER_TESTS "Build the tests" ON)
IF(DSFILTER_TESTS)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF()

# generate and install pkg-config file
# (holy crap this is easier than a .cmake)
SET(DSFILTER_PC_NAME dsfilter-${DSFILTER_VERSION_MAJOR}.${DSFILTER_VERSION_MINOR}.pc)
CONFIGURE_FILE(dsfilter.pc.in ${DSFILTER_PC_NAME} @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${DSFILTER_PC_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)

# install library and headers
INSTALL(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib/${CMAKE_CXX_LIBRARY_ARCHITECTURE}
  PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)
