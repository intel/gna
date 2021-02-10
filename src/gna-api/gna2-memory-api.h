/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

/**************************************************************************//**
 @file gna2-memory-api.h
 @brief Gaussian and Neural Accelerator (GNA) 2.0 API Definition.
 @nosubgrouping

 ******************************************************************************
 @addtogroup GNA2_API
 @{

 ******************************************************************************

 @addtogroup GNA2_API_MEMORY Memory API

 API for managing memory used by GNA Tensors.

 @{
 *****************************************************************************/

#ifndef __GNA2_MEMORY_API_H
#define __GNA2_MEMORY_API_H

#include "gna2-common-api.h"

#include <stdint.h>

/**
 Allocates memory buffer, that can be used with GNA device.

 @param sizeRequested Buffer size desired by the caller. Must be within range <1, 2^28>.
 @param [out] sizeGranted Buffer size granted by GNA,
                      can be more then requested due to HW constraints.
 @param [out] memoryAddress Address of memory buffer
 @return Status of the operation.
    @retval Gna2StatusSuccess On success.
    @retval Gna2StatusMemorySizeInvalid If sizeRequested is invalid.
 */
GNA2_API enum Gna2Status Gna2MemoryAlloc(
    uint32_t sizeRequested,
    uint32_t * sizeGranted,
    void ** memoryAddress);

/**
 Releases memory buffer.

 @param memory Memory buffer to be freed.
 @return Status of the operation.
 */
GNA2_API enum Gna2Status Gna2MemoryFree(
    void * memory);

#endif // __GNA2_MEMORY_API_H

/**
 @}
 @}
 */
