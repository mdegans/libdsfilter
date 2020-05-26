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

FROM nvcr.io/nvidia/deepstream:5.0-dp-20.04-devel

ARG VERSION="UNSET - check docker_build.sh"
ARG BUILD_DIR="/usr/local/src/dsfilter"

# fix deepstream permissions and ldconfig
RUN chmod -R o-w /opt/nvidia/deepstream/deepstream-5.0/ \
    && echo "/opt/nvidia/deepstream/deepstream/lib" > /etc/ld.so.conf.d/deepstream.conf \
    && ldconfig

WORKDIR ${BUILD_DIR}
COPY CMakeLists.txt dsfilter.pc.in DsfilterConfig.cmake.in LICENSE README.md ./
COPY src ./src/
COPY include ./include/
COPY test ./test/

# install deps and create user
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libprotobuf-dev \
        cmake \
        ninja-build \
    && mkdir build \
    && cd build \
    && cmake -GNinja .. \
    && ninja \
    && ninja install \
    && apt-get purge -y --autoremove \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*
