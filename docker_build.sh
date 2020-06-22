#!/bin/bash

set -ex

# change this if you fork the repo and want to push you own image
readonly AUTHOR="mdegans"
readonly PROJ_NAME="libdsfilter"

# change this if you want to override the architecture (cross build)
# this is untested and unsupported
readonly ARCH="$(arch)"

TAG_SUFFIX=$(git rev-parse --abbrev-ref HEAD)
if [[ $TAG_SUFFIX == "master" ]]; then
    TAG_SUFFIX="latest"
fi
readonly DOCKERFILE_BASENAME="latest.Dockerfile"

# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
readonly DOCKERFILE="$THIS_DIR/$DOCKERFILE_BASENAME"
readonly DISTANCEPROT0_VERSION="0.3.1"
readonly VERSION=$(head -n 1 $THIS_DIR/VERSION)
readonly TAG_BASE="$AUTHOR/$PROJ_NAME"
TAG_FULL="$TAG_BASE:$VERSION"


readonly DISTANCEPROTO_TAG="$DISTANCEPROT0_VERSION-ds-$ARCH"
readonly TAG_SUFFIX="${TAG_SUFFIX}-$ARCH"
readonly TAG_FULL="${TAG_FULL}-$ARCH"

echo "Building $TAG_FULL from $DOCKERFILE"

docker build --pull --rm -f $DOCKERFILE \
    --build-arg DISTANCEPROTO_TAG="$DISTANCEPROTO_TAG" \
    -t $TAG_FULL \
    $THIS_DIR $@
docker tag "$TAG_FULL" "$TAG_BASE:$TAG_SUFFIX"
