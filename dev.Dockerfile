# This Dockerfile is for making rapid changes and testing

# Copyright (C) 2020  Michael de Gans

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

ARG DISTANCEPROTO_TAG="UNSET (use docker_build.sh to build)"
FROM mdegans/libdistanceproto:${DISTANCEPROTO_TAG}

ARG BUILD_DIR="/usr/local/src/dsfilter"

# install deps and create user
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libprotobuf-dev \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# copy source
WORKDIR ${BUILD_DIR}
COPY CMakeLists.txt dsfilter.pc.in DsfilterConfig.cmake.in VERSION LICENSE README.md ./
COPY src ./src/
COPY include ./include/
COPY test ./test/

# build and install
RUN mkdir build \
    && cd build \
    && cmake -GNinja .. \
    && ninja \
    && ninja install \
    && ldconfig \
    && cd .. \
    && rm -rf build

# verbose gstreamer logging
ENV GST_DEBUG="4"
ENV G_MESSAGES_DEBUG="all"
