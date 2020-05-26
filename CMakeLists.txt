PROJECT(dsfilter CXX CUDA)

SET(DSFILTER_VERSION_MAJOR 0)
SET(DSFILTER_VERSION_MINOR 1)
SET(DSFILTER_VERSION_PATCH 0)
SET(DSFILTER_VERSION_STRING ${DSFILTER_VERSION_MAJOR}.${DSFILTER_VERSION_MINOR}.${DSFILTER_VERSION_PATCH})

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GSTREAMER REQUIRED gstreamer-1.0)

INCLUDE_DIRECTORIES(${GSTREAMER_INCLUDE_DIRS})
# INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# deepstream stuff
LINK_DIRECTORIES(/opt/nvidia/deepstream/deepstream/lib/)
# TARGET_LINK_DIRECTORIES doesn't seem to be supported in cmake 3.10\
# if anybody knows how to add this to the target so it can be exported
# it would be appreciated. My cmake is meh.

ADD_LIBRARY(${PROJECT_NAME} SHARED
    src/BaseCudaFilter.cu
    src/TestCudaFilter.cu
    src/BaseFilter.cpp
    src/distance.pb.cc
    src/HashCudaFilter.cpp
)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES VERSION ${DSFILTER_VERSION_STRING} SOVERSION ${DSFILTER_VERSION_MAJOR})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "include/BaseFilter.hpp;include/DistanceFilter.hpp;include/BaseCudaFilter.hpp;include/utils.hpp")

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME};>
  /opt/nvidia/deepstream/deepstream/sources/includes
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${GSTREAMER_LIBRARIES} nvds_meta nvdsgst_meta)

# generate and install pkg-config file
CONFIGURE_FILE(dsfilter.pc.in dsfilter.pc @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/dsfilter.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)

# install library and headers
INSTALL(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
    LIBRARY DESTINATION lib/${CMAKE_CXX_LIBRARY_ARCHITECTURE}
    PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)

# generating a .cmake file is positively arcane
# but if anybody can complete this part it would be nice and 
# save me hours of going through the cmake docs
