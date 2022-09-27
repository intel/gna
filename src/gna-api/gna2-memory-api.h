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
 For Linux OS, following is in place:
 1. when score request is enqueued to given GNA HW, it's memory buffers must be bound to that GNA HW.
 2. buffer to GNA HW binding can be done in two ways:
    1. (Implicitly) through Gna2MemoryAlloc(...) - first already opened GNA HW will be chosen (if any). For most cases GNA HW is opened prior any subsequent operation.
    2. (Explicitly) through Gna2MemoryAllocForDevice(deviceIndex, ...) with deviceIndex pointing to that device - useful when more than one GNA HW is present in system.
 3. memory rebinding is impossible.
 4. when GNA HW is closed, all memory buffers bound to it are also implicitly closed.
 */


/**
 Allocates memory buffer, that can be used with GNA device.

 see comment above

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
 Allocates memory buffer, that can be used with GNA device - device aware

 see comment above

 @param deviceIndex assures device is open prior allocation.
 @param sizeRequested Buffer size desired by the caller. Must be within range <1, 2^28>.
 @param [out] sizeGranted Buffer size granted by GNA,
                          can be more then requested due to HW constraints.
 @param [out] memoryAddress Address of memory buffer
 @return Status of the operation.
    @retval Gna2StatusSuccess On success.
    @retval Gna2StatusMemorySizeInvalid If sizeRequested is invalid.

 */
GNA2_API enum Gna2Status Gna2MemoryAllocForDevice(
    uint32_t deviceIndex,
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
