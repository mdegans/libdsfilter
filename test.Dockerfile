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

FROM nvcr.io/nvidia/deepstream:4.0.2-19.12-devel

ARG PROJECT_NAME="dsfilter"
ARG DSFILTER_USERNAME="${PROJECT_NAME}"
ARG DSFILTER_HOME="/var/${DSFILTER_USERNAME}"
ARG DSFILTER_PREFIX="${DSFILTER_HOME}/.local/"
ARG DSFILTER_SRCDIR="${DSFILTER_PREFIX}/src/dsfilter"

# install deps and create user
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libglib2.0-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        ninja-build \
    && pip3 install meson \
    && apt-get purge -y --autoremove \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && cp -R /root/deepstream_sdk_v4.0.2_x86_64/sources/ /opt/nvidia/deepstream/deepstream-4.0/ \
    && useradd -md ${DSFILTER_HOME} -rUs /bin/false ${DSFILTER_USERNAME} \
    && mkdir -p ${DSFILTER_SRCDIR} \
    && chown -R ${DSFILTER_USERNAME}:${DSFILTER_USERNAME} ${DSFILTER_HOME}
# mkdir and chown because WORKDIR doesn't respect USER or have a --chown flag because excuses
# https://github.com/moby/moby/issues/20295#issuecomment-254951672

# drop to user
USER ${DSFILTER_USERNAME}:${DSFILTER_USERNAME}

# copy source
WORKDIR ${DSFILTER_SRCDIR}
COPY meson.build ./
COPY src ./src/
COPY include ./include/
COPY config ./config/

# build and install to user home
RUN mkdir build \
    && cd build \
    && meson --prefix ${DSFILTER_PREFIX} .. \
    && ninja \
    && ninja test \
    && ninja install \
    && cd .. \
    && rm -rf build

# verbose gstreamer logging
ENV GST_DEBUG="4"
ENV G_MESSAGES_DEBUG="all"
