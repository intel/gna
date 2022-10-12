/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "KernelArguments.h"

#include <cstdint>

struct GmmConfig;

typedef void (*GmmMaxMix)(ExecutionKernelConfig<GmmConfig> const * const config);

typedef void (*GmmMaxMixActiveList)(
        ExecutionKernelConfig<GmmConfig> const * const config, AffineConfigAl al);

struct GmmKernel
{
    GmmMaxMix gmmMaxMix8;
    GmmMaxMix gmmMaxMix16;
    GmmMaxMixActiveList gmmMaxMix8ActiveList;
    GmmMaxMixActiveList gmmMaxMix16ActiveList;
} ;

// Export list of available GMM kernels providers

// generic GMM kernel provider
extern GmmKernel gmmKernel_generic;

// sse4.2 accelerated GMM kernel provider
extern GmmKernel gmmKernel_sse4;

// avx1 accelerated GMM kernel provider
extern GmmKernel gmmKernel_avx1;

// avx2 accelerated GMM kernel provider
extern GmmKernel gmmKernel_avx2;
