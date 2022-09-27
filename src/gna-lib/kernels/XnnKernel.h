/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "KernelArguments.h"
#include "ConvolutionKernelArguments.h"
#include "PoolingKernelArguments.h"
#include "pwl.h"

#include "../gna-api/gna2-inference-impl.h"

#include <map>

struct ActivationConfig;
struct AffineConfig;
struct AffineConfigAl;
struct ConvolutionConfig2D;
struct ConvolutionConfig;
struct CopyConfig;
struct PoolingConfig2D;
struct PoolingConfig;
struct RecurrentConfig;
struct TransposeConfig;
template <typename TransformConfig> struct ExecutionKernelConfig;

namespace GNA
{
struct PwlCached;

template<typename KernelType>
using KernelMap = std::map<AccelerationMode, KernelType>;

typedef void (*VoidKernel)();

typedef void (*AffineKernel)(ExecutionKernelConfig<AffineConfig> const * const config);

typedef void (*AffineActiveListKernel)(
        ExecutionKernelConfig<AffineConfig> const * const config, AffineConfigAl al);

typedef void (*ActivationKernel)(ExecutionKernelConfig<ActivationConfig> const * const config);

typedef void (*RecurrentKernel)(ExecutionKernelConfig<RecurrentConfig> const * const config);

typedef void (*ConvolutionKernel)(ConvolutionConfig const * const config);

typedef void (*ConvolutionKernel2D)(ExecutionKernelConfig<ConvolutionConfig2D> const * const config);

typedef void (*PoolingKernel2D)(ExecutionKernelConfig<PoolingConfig2D> const * const config);

typedef void (*ConvolutionPoolingKernel)(ConvolutionConfig const * const filterConfig,
    PoolingConfig const * const poolConfig, PwlCached const * const pwl);

typedef void (*TransposeKernel)(TransposeConfig const * const config);

typedef void (*CopyKernel)(CopyConfig const * const config);

enum KernelType
{
    affineSingle1Bfull,
    affineSingle2Bfull,
    affineSingle1Bal,
    affineSingle2Bal,
    affineMulti1B,
    affineMulti2B,
    diagonal1B,
    diagonal2B,
    recurrent1B,
    recurrent2B,
    transpose1B,
    transpose2B,
    convolution,
    convolutionPooling,
    pwl,
    copy,
    affineSingle1B1Bfull,
    affineSingle2B1Bfull,
    affineSingle1B2Bfull,
    affineSingle2B2Bfull,
    affineSingle1B1Bal,
    affineSingle2B1Bal,
    affineSingle1B2Bal,
    affineSingle2B2Bal,
    affineMulti1B1B,
    affineMulti2B1B,
    affineMulti1B2B,
    affineMulti2B2B,
    diagonal1B1B,
    diagonal2B1B,
    diagonal1B2B,
    diagonal2B2B,
    recurrent1B1B,
    recurrent2B1B,
    recurrent1B2B,
    recurrent2B2B,
    convolution1B,
    convolutionPooling1B,
    convolution2B,
    convolutionPooling2B,
    copy1B,
    copy2B,
    convolution2D1B1B,
    convolution2D1B2B,
    convolution2D2B1B,
    convolution2D2B2B,
    convolutionPooling2D1B,
    convolutionPooling2D2B,
    convolutionPooling2D4B,
};

template<Gna2AccelerationMode accelerationMode>
VoidKernel GetXnnKernel(KernelType type);

void setHwCompatibilityMode_generic(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_sse4(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx1(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx2(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_generic_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_sse4_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx1_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);
void setHwCompatibilityMode_avx2_sat(uint32_t bufferElementCounts[2][XNN_N_GROUP_MAX]);

}
