/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GNA_API_H
#define __GNA_API_H

#include <stdint.h>

#if !defined(_WIN32)
#include <assert.h>
#endif

#include "gna-api-status.h"
#include "gna-api-types-gmm.h"
#include "gna-api-types-xnn.h"

/******************************************************************************
 * GNA Capabilities API
 *****************************************************************************/
typedef enum _api_version
{
    GNA_API_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
    GNA_API_1_0,
    GNA_API_2_0,
    GNA_API_3_0,

    GNA_API_VERSION_COUNT
} gna_api_version;

typedef enum _device_generation
{
    GNA_DEVICE_NOT_SUPPORTED = (int)GNA_NOT_SUPPORTED,
    GMM_DEVICE,
    GNA_0_9,
    GNA_1_0,
    GNA_1_0_EMBEDDED,
    GNA_2_0,
    GNA_3_0,
    GNA_DEVICE_COUNT
} gna_device_generation;

/**
 *  Enumeration of device flavors
 */
typedef enum _gna_device_version
{
    GNA_UNSUPPORTED  = (int)GNA_NOT_SUPPORTED, // No supported hardware device available.
    GNA_GMM          = 0x01,
    GNA_0x9          = 0x09,
    GNA_1x0          = 0x10,
    GNA_2x0          = 0x20,
    GNA_EMBEDDED_1x0 = 0x10E,
    GNA_SOFTWARE_EMULATION = GNA_DEFAULT, // Software emulation fall-back will be used.

} gna_device_version;


/** Maximum number of requests that can be enqueued before retrieval */
const uint32_t GNA_REQUEST_QUEUE_LENGTH = 64;

/******************************************************************************
 * GNA Utilities API
 *****************************************************************************/

/**
 * Rounds a number up, to the nearest multiple of significance
 * Used for calculating the memory sizes of GNA data buffers
 *
 * @param number        Memory size or a number to round up.
 * @param significance  Informs the function how to round up. The function "ceils"
 *                      the number to the lowest possible value divisible by "significance".
 * @return Rounded integer value.
 * @deprecated          Will be removed in next release.
 */
#define ALIGN(number, significance)   ((((number) + (significance) - 1) / (significance)) * (significance))

/**
 * Rounds a number up, to the nearest multiple of 64
 * Used for calculating memory sizes of GNA data arrays
 * @deprecated          Will be removed in next release.
 */
#define ALIGN64(number)   ALIGN(number, 64)

/**
 * Verifies data sizes used in the API and GNA hardware
 *
 * NOTE: If data sizes in an application using API differ from data sizes
 *       in the API library implementation, scoring will not work properly
 */
static_assert(1 == sizeof(int8_t), "Invalid size of int8_t");
static_assert(2 == sizeof(int16_t), "Invalid size of int16_t");
static_assert(4 == sizeof(int32_t), "Invalid size of int32_t");
static_assert(1 == sizeof(uint8_t), "Invalid size of uint8_t");
static_assert(2 == sizeof(uint16_t), "Invalid size of uint16_t");
static_assert(4 == sizeof(uint32_t), "Invalid size of uint32_t");

#endif  // ifndef __GNA_API_H
