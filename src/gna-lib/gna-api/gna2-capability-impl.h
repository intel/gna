/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GNA2_CAPABILITY_IMPL_H
#define __GNA2_CAPABILITY_IMPL_H

#include "gna2-capability-api.h"

#include "gna2-common-impl.h"

#include <stdint.h>

namespace GNA
{

typedef enum Gna2DeviceGeneration DeviceGeneration;


/**
 Generation of device that is used by default by GNA Library in software mode,
 when no hardware device is available.

 @see
 Gna2DeviceVersion.

 @note
 Usually it will be the latest existing GNA generation (excluding embedded)
 on the time of publishing the library, value may change with new release.
 */
#define GNA2_DEFAULT_DEVICE_GENERATION Gna2DeviceGeneration2_0

DeviceGeneration const DefaultDeviceGeneration = GNA2_DEFAULT_DEVICE_GENERATION;

}

#endif // __GNA2_CAPABILITY_IMPL_H
