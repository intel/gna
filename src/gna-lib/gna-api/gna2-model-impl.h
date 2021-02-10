/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GNA2_MODEL_IMPL_H
#define __GNA2_MODEL_IMPL_H

#include "gna2-model-api.h"

namespace GNA
{

typedef struct Gna2Model ApiModel;
typedef struct Gna2Operation ApiOperation;
typedef struct Gna2Shape ApiShape;
typedef struct Gna2Tensor ApiTensor;
typedef struct Gna2ModelError ModelError;

typedef enum Gna2OperationType OperationType;
typedef enum Gna2TensorMode TensorMode;
typedef enum Gna2DataType DataType;
typedef enum Gna2BiasMode ApiBiasMode;
typedef enum Gna2PoolingMode Mode;
typedef enum Gna2ErrorType ErrorType;
typedef enum Gna2ItemType ItemType;

constexpr uint32_t ScratchpadOperandIndex = UINT32_MAX;
constexpr uint32_t InputOperandIndex = 0;
constexpr uint32_t OutputOperandIndex = 1;
constexpr uint32_t WeightOperandIndex = 2;
constexpr uint32_t FilterOperandIndex = 2;
constexpr uint32_t BiasOperandIndex = 3;
constexpr uint32_t PwlOperandIndex = 4;
constexpr uint32_t WeightScaleFactorOperandIndex = 5;
constexpr uint32_t GmmInterleavedOperandIndex = 2;
constexpr uint32_t GmmMeanOperandIndex = 2;
constexpr uint32_t GmmInverseCovarianceOperandIndex = 3;
constexpr uint32_t GmmGaussianConstantOperandIndex = 4;

// NOTE: temporary solution for simple and fast kernel buffer indexing, always set as last + 1 operand index
constexpr uint32_t ScratchpadOperandKernelIndex = 6;

// NOTE: helper for calculating SW-only scratchpad, currently only used by cnnFusedBuffer size
constexpr uint32_t SoftwareScratchpadOperandIndex = 7;


constexpr uint32_t ConvolutionStrideParamIndex = 0;
constexpr uint32_t BiasModeConvolutionParamIndex = 1;
constexpr uint32_t PoolingModeParamIndex = 2;
constexpr uint32_t PoolingWindowParamIndex = 3;
constexpr uint32_t PoolingStrideParamIndex = 4;
constexpr uint32_t ZeroPaddingParamIndex = 5;

constexpr uint32_t BiasModeAffineParamIndex = 0;
constexpr uint32_t BiasVectorParamIndex = 1;

}

#endif // __GNA2_MODEL_IMPL_H
