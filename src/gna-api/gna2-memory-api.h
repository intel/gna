/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

/**************************************************************************//**
 @file gna2-memory-api.h
 @brief Gaussian and Neural Accelerator (GNA) 3.0 API Definition.
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

#include <cstdint>

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

/**
 Adds special designation for the memory buffer.
 The buffer can be one of the previously obtained with Gna2MemoryAlloc.
 @note
 - This is for embedded model export only.
 - In case of some doubts, probably should not use this function.

 @param memory Starting address of the memory buffer to tag.
 @param tag Special purpose tag. Use zero to reset to default. @see ::Gna2MemoryTag
 @return Status of the operation.
    @retval Gna2StatusSuccess On success.
    @retval Gna2StatusMemoryBufferInvalid If memory address is invalid.
 */
GNA2_API enum Gna2Status Gna2MemorySetTag(
    void * memory,
    uint32_t tag);


#endif // __GNA2_MEMORY_API_H

/**
 @}
 @}
 */
