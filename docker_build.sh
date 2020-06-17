#!/bin/bash

set -ex

# change this if you fork the repo and want to push you own image
readonly AUTHOR="mdegans"
readonly PROJ_NAME="libdsfilter"

TAG_SUFFIX=$(git rev-parse --abbrev-ref HEAD)
if [[ $TAG_SUFFIX == "master" ]]; then
    TAG_SUFFIX="latest"
fi
readonly DOCKERFILE_BASENAME="latest.Dockerfile"

# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
readonly DOCKERFILE="$THIS_DIR/$DOCKERFILE_BASENAME"
readonly VERSION=$(head -n 1 $THIS_DIR/VERSION)
readonly TAG_BASE="$AUTHOR/$PROJ_NAME"
TAG_FULL="$TAG_BASE:$VERSION"

# TODO(mdegans): using deepstream-$(arch) may be cleaner
if [[ "$(arch)" == "aarch64" ]]; then
    readonly DISTANCEPROTO_TAG="deepstream-tegra"
    readonly TAG_SUFFIX="${TAG_SUFFIX}-tegra"
    readonly TAG_FULL="${TAG_FULL}-tegra"
else
    readonly DISTANCEPROTO_TAG="deepstream-x86"
    readonly TAG_SUFFIX="${TAG_SUFFIX}-x86"
    readonly TAG_FULL="${TAG_FULL}-x86"
fi

echo "Building $TAG_FULL from $DOCKERFILE"

docker build --pull --rm -f $DOCKERFILE \
    --build-arg DISTANCEPROTO_TAG="$DISTANCEPROTO_TAG" \
    -t $TAG_FULL \
    $THIS_DIR $@
docker tag "$TAG_FULL" "$TAG_BASE:$TAG_SUFFIX"
