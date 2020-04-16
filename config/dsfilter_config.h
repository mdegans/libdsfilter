#ifndef __DSFILTER_CONFIG_H__
#define __DSFILTER_CONFIG_H__

#pragma once

#ifdef HAVE_CONFIG_H
#include "dsfilter_config_meson.h"
#define DSFILTER_BUILD_SYSTEM "meson"
#else
#include "dsfilter_config_dev.h"
#define DSFILTER_BUILD_SYSTEM "dev"
#endif

#endif