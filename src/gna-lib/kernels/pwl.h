/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/


#pragma once

#include "KernelArguments.h"

#include <cstdint>

template<typename TransformConfig>
struct ExecutionKernelConfig;

struct ActivationConfig;

// PWL Segment x base type
typedef int64_t pwl_x_t;

// Unpacked double-segment for lookup table entry
typedef struct
{
    pwl_x_t     xBaseA;
    pwl_x_t     xBaseB;
    int16_t     slopeA;
    int16_t     shiftA;
    int16_t     reservedA;
    int16_t     yBaseA;
    int16_t     slopeB;
    int16_t     shiftB;
    int16_t     reservedB;
    int16_t     yBaseB;
} pwl_u_t;

static_assert(32 == sizeof(pwl_u_t), "Invalid size of pwl_u_t");

// PWL Unpacked single-segment, auxiliary for lookup preparation
typedef struct
{
    pwl_x_t     xBase;
    int16_t     slope;
    int16_t     shift;
    int16_t     resvd;
    int16_t     yBase;
} pwl_s_t;

static_assert(16 == sizeof(pwl_s_t), "Invalid size of pwl_s_t");

// PWL Unpacked segment values, for split PWL segment and binary search
typedef struct __pwl_y
{
    int16_t     slope;
    int16_t     shift;
    int16_t     resvd;
    int16_t     yBase;
} pwl_y_t;

static_assert(8 == sizeof(pwl_y_t), "Invalid size of pwl_y_t");

namespace GNA
{

// PWL cached config (constant for given layer)
struct PwlCachedConfig
{
    uint32_t segmentCount;
    uint32_t bytesPerOutput;
    void * data;

    union PwlCachedParams
    {
        struct
        {
            pwl_x_t xBase0;             // first segment xBase value (Lookup algorithm)
            pwl_x_t xBase0Neg;          // first segment xBase value x -1 for addition only  (Lookup algorithm)
            pwl_x_t xBase1diff;         // difference between first and second PWL segment xBases, for lookup
            int16_t slope0;             // first segment slope value (Lookup algorithm)
            int16_t shift0;             // first segment extracted shift value (Lookup algorithm)
            int16_t yBase0;             // first segment yBase value (Lookup algorithm)
            uint16_t count;             // number of lookup segments (active)
            uint8_t width;
            uint8_t _reserved[7];       // padding
        } Lookup;
        struct
        {
            PwlSegment* source;         // unpacked segments
            pwl_y_t*  ySeg;             // extracted PWL segments value data
            pwl_x_t xBase0;             // first segment xBase value (binary search algorithm)
            int16_t yBase0;             // first segment yBase value (binary search algorithm)
            uint8_t shift0;             // first segment slope_shift
            uint8_t _reserved[5];       // padding
        } Binary;
    } Params;
};

// Function pointer for apply PWL for single input-output
typedef void(*PwlApplySingle)(PwlCachedConfig const * const pwl, int32_t I, int16_t * const output,
        uint32_t * const saturationCount);

// Function pointer for apply PWL for all inputs-outputs
typedef void(*PwlApplyAll)(ExecutionKernelConfig<ActivationConfig> const * const config);

// PWL cache and config (constant for given layer)
struct PwlCached
{
    bool useLookup = false;

    void InitializeActivationFunctions_generic_sat() const;
    void InitializeActivationFunctions_sse4_sat() const;
    void InitializeActivationFunctions_avx1_sat() const;
    void InitializeActivationFunctions_avx2_sat() const;

    // Prepares PWL parameters and auxiliary buffers
    PwlCached(uint32_t elementSize, PwlSegment const * const segmentsIn, uint32_t segmentCountIn);
    PwlCached(PwlCached&& pwlCached) = delete;
    PwlCached(const PwlCached& pwlCached) = delete;
    ~PwlCached();

    // PWL LOOKUP table number of elements
    static const int32_t PWL_LOOKUP_COUNT = 1024;

    // PWL LOOKUP table element size in B
    static const int32_t PWL_LOOKUP_SEG_SIZE = sizeof(pwl_u_t);

    // PWL LOOKUP table - number of segments per element
    static const int32_t PWL_LOOKUP_SEG_SCOUNT = 2;

    // PWL LOOKUP table size in B
    static const int32_t PWL_LOOKUP_SIZE = (PWL_LOOKUP_COUNT)* PWL_LOOKUP_SEG_SIZE;

    PwlCachedConfig pwl;
    mutable PwlApplySingle  ActivateSingle;              // algorithm used for PWL for single in-out
    mutable PwlApplyAll     ActivateAll;                 // algorithm used for PWL for all in-outs

private:
    void allocateLookupCaches();
    void allocateBinaryCaches();
};

}
