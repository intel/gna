/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

/**************************************************************************//**
 @file gna2-capability-api.h
 @brief Gaussian and Neural Accelerator (GNA) 3.0 API Definition.
 @nosubgrouping

 ******************************************************************************

 @addtogroup GNA2_API
 @{
 ******************************************************************************

 @addtogroup GNA2_CAPABILITY_API Capability API

 API for querying capabilities of hardware devices and library.

 @{
 *****************************************************************************/

#ifndef __GNA2_CAPABILITY_API_H
#define __GNA2_CAPABILITY_API_H

#include "gna2-common-api.h"

/**
 List of device generations.

 Generation determines the set of capabilities common amongst multiple device versions.
 @see Gna2DeviceVersion.
 */
enum Gna2DeviceGeneration
{
    /**
     Legacy device supporting only Gaussian Mixture Models scoring.
     */
    Gna2DeviceGenerationGmm = 0x010,

    /**
     Initial GNA device generation with no CNN support.
     Backward compatible with ::Gna2DeviceGenerationGmm.
     */
    Gna2DeviceGeneration0_9 = 0x090,

    /**
     First fully featured GNA device generation.
     Backward compatible with ::Gna2DeviceGeneration0_9.
     */
    Gna2DeviceGeneration1_0 = 0x100,

    /**
     Fully featured second GNA device generation.
     Backward compatible with ::Gna2DeviceGeneration1_0.
     */
    Gna2DeviceGeneration2_0 = 0x200,

    /**
     Fully featured third GNA device generation.
     Partially compatible with ::Gna2DeviceGeneration2_0.
     */
    Gna2DeviceGeneration3_0 = 0x300,

    /**
     2D CNN enhanced third GNA device generation.
     Partially compatible with ::Gna2DeviceGeneration2_0.
     Fully compatible with ::Gna2DeviceGeneration3_0.
     */
    Gna2DeviceGeneration3_1 = 0x310,
};

/**
 Gets library version string.

 @param [out] versionBuffer User allocated buffer for the version string
 @param [in] versionBufferSize The size of the versionBuffer in bytes.
        Must be at least 32 bytes long.
 @return Status of fetching the version.
    @retval Gna2StatusSuccess The version was fully serialized into the versionBuffer.
    @retval Gna2StatusMemorySizeInvalid The versionBuffer is too small.
    @retval Gna2StatusNullArgumentNotAllowed The versionBuffer was NULL.
 */
GNA2_API enum Gna2Status Gna2GetLibraryVersion(char * versionBuffer, uint32_t versionBufferSize);

#endif // __GNA2_CAPABILITY_API_H

/**
 @}
 @}
 */
