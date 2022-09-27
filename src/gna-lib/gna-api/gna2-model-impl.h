/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#ifndef __GNA2_MODEL_IMPL_H
#define __GNA2_MODEL_IMPL_H

#include "gna2-model-api.h"

#include <initializer_list>

namespace GNA
{

typedef struct Gna2Model ApiModel;
typedef struct Gna2Shape ApiShape;
typedef struct Gna2Tensor ApiTensor;
typedef struct Gna2PwlSegment PwlSegment;
typedef struct Gna2WeightScaleFactor WeightScaleFactor;
typedef struct Gna2CompoundBias BiasCompound;
/** Bias (constant) data type */
typedef int32_t BiasRegular;

typedef enum Gna2OperationType OperationType;
typedef enum Gna2TensorMode TensorMode;
typedef enum Gna2DataType DataType;
typedef enum Gna2BiasMode ApiBiasMode;
typedef enum Gna2PoolingMode Mode;

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

constexpr uint32_t CopyShapeParamIndex = 0;
constexpr uint32_t ConvolutionStrideParamIndex = 0;
constexpr uint32_t BiasModeConvolutionParamIndex = 1;
constexpr uint32_t PoolingModeParamIndex = 2;
constexpr uint32_t PoolingWindowParamIndex = 3;
constexpr uint32_t PoolingStrideParamIndex = 4;
constexpr uint32_t ZeroPaddingParamIndex = 5;

constexpr uint32_t BiasModeAffineParamIndex = 0;
constexpr uint32_t BiasVectorParamIndex = 1;

constexpr uint32_t ThresholdConditionParamIndex = 0;
constexpr uint32_t ThresholdModeParamIndex = 1;
constexpr uint32_t ThresholdMaskParamIndex = 2;

typedef enum _TransformOperation
{
    ActivationTransform,
    AffineTransform,
    AffineDiagonalTransform,
    AffineMultibiasTransform,
    ConvolutionalTransform1D,
    ConvolutionalTransform2D,
    CopyTransform,
    TransposeTransform,
    GmmTransform,
    PoolingTransform1D,
    PoolingTransform2D,
    RecurrentTransform,
    TransformOperationCount,
} TransformOperation;

/******************************************************************************
 * GNA Enumerations
 *****************************************************************************/

/**
 * Order of data tensor used by inputs, outputs, biases, weights etc.
 * Used as (OR'ed) binary flags for reporting capabilities, e.g. (GNA_TENSOR_NHWD | GNA_TENSOR_NDHW);
 */
typedef enum _tensor_order
{
    GNA_TENSOR_SCALAR = 0,      // Scalar, order = 0
    GNA_TENSOR_W = 1,           // Width (1D vector)
    GNA_TENSOR_H = 2,           // Height (1D vector)
    GNA_TENSOR_NW = 4,          // Grouping, Width (2D Matrix) AKA INTERLEAVED
    GNA_TENSOR_NH = 8,          // Grouping, Height (2D Matrix) AKA INTERLEAVED
    GNA_TENSOR_WN = 16,         // Width, Grouping (2D Matrix) AKA DEINTERLEAVED/FLAT
    GNA_TENSOR_WH = 32,         // Width, Height (2D Matrix) (Weights)
    GNA_TENSOR_HN = 64,         // Height, Grouping (2D Matrix) AKA DEINTERLEAVED/FLAT
    GNA_TENSOR_HW = 128,        // Height, Width (2D Matrix) common for all 2D tensors
    GNA_TENSOR_HD = 256,        // Height, Depth/Channel, (2D Matrix)
    GNA_TENSOR_HDW = 512,       // Height, Depth/Channel, Width (3D Tensor)
    GNA_TENSOR_NWH = 1024,      // Grouping, Width, Height (3D Tensor)
    GNA_TENSOR_WHD = 2048,      //  Width, Height, Depth/Channel (3D Tensor)
    GNA_TENSOR_NHWD = 4096,     // N -Grouping[=1]/Number of filters, Height, Width, Depth/Channel (GNA 2D CNN default) (4D Tensor)
    GNA_TENSOR_NDHW = 8192,     // N -Grouping[=1]/Number of filters, Depth/Channel, Height, Width, (TensorFlow) (4D Tensor)
    GNA_TENSOR_ORDER_ANY = -1,  // ordering as in gna_tensor_dim beginning with GNA_DIM_N
    GNA_TENSOR_NHW = 4097,     // Temporary value for Bias Shape
    GNA_TENSOR_N = 3,     // Temporary value for Bias Shape
    GNA_TENSOR_HWD = 513,
    GNA_TENSOR_NWD = 514,       // Used for Legacy Convolution output
} gna_tensor_order;

/**
 * Helper Tensor dimension selector for dimension map.
 */
typedef enum _tensor_dim
{
    GNA_DIM_S,          // Scalar
    GNA_DIM_N,          // Grouping (Batch size)
    GNA_DIM_W,          // Width
    GNA_DIM_H,          // Height
    GNA_DIM_D,          // Depth (for 2D operations same as Channel)
    //GNA_DIM_C,          // Channel (for 2D operations same as Depth)

    GNA_DIM_X,
    GNA_DIM_Y,
    GNA_DIM_Z,
} gna_tensor_dim;

/**
 * Layer operation type.
 * Defines type of layer "core" operation.
 * All nodes/cells within a layer are of the same type,
 * e.g. affine transform cell, convolutional cell, recurrent cell.
 * Affine, convolutional and recurrent layers are in fact "fused operation" layers
 * and "core" operation is fused with activation and/or pooling functions.
 * NOTE: Operation types are exclusive.
 */
typedef enum _layer_operation
{
    INTEL_AFFINE,                   // Fully connected affine transform (deep feed forward) with activation function. Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_DIAGONAL,          // Fully connected affine transform (matrix x vector) (deep feed forward) with activation function.Cast pLayerStruct to intel_affine_layer_t.
    INTEL_AFFINE_MULTIBIAS,         // Fully connected affine transform (with grouped bias vectors) (deep feed forward) with activation function. Cast pLayerStruct to intel_affine_multibias_layer_t.
    INTEL_CONVOLUTIONAL,            // Convolutional transform with activation function and pooling. Cast pLayerStruct to intel_convolutional_layer_t.
    INTEL_CONVOLUTIONAL_2D,         // Convolutional transform with activation function and pooling. Cast pLayerStruct to nn_layer_cnn2d.
    INTEL_CONVOLUTIONAL_1D,         // FOR INTERNAL USE ONLY
    INTEL_COPY,                     // Auxiliary data copy operation. Cast pLayerStruct to intel_copy_layer_t.
    INTEL_DEINTERLEAVE,             // Auxiliary 2D tensor transpose operation (interleave to flat). No casting, always set pLayerStruct to null.
    INTEL_GMM,                      // Gaussian Mixture Model operation. Cast pLayerStruct to intel_gmm_layer_t.
    INTEL_INTERLEAVE,               // Auxiliary 2D tensor transpose operation (flat to interleave). No casting, always set pLayerStruct to null.
    INTEL_RECURRENT,                // Fully connected affine transform with recurrence and activation function. Cast pLayerStruct to intel_recurrent_layer_t.
    GNA_LAYER_CNN_2D_POOLING,
    INTEL_AFFINE_THRESHOLD,
    LAYER_OPERATION_TYPE_COUT,      // Number of Layer operation types.
} nn_operation;

}

#endif // __GNA2_MODEL_IMPL_H
