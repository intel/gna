/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Bias.h"

#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "PoolingFunctions2D.h"
#include "Shape.h"
#include "Validator.h"

#include "ConvolutionKernelArguments.h"

#include "gna2-common-api.h"
#include "common.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <cstdint>
#include <map>
#include <memory>
#include <utility>

using namespace GNA;

const FullCapabilitiesMap BiasTensor::capabilities =
{
    {INTEL_AFFINE, {
        AffineLayerCapabilities::GetOperands(BiasOperandIndex).at(INTEL_AFFINE)
    }},
    {INTEL_AFFINE_DIAGONAL, {
        AffineLayerCapabilities::GetOperands(BiasOperandIndex).at(INTEL_AFFINE_DIAGONAL)
    }},
    {INTEL_AFFINE_MULTIBIAS, {
        AffineLayerCapabilities::GetOperands(BiasOperandIndex).at(INTEL_AFFINE_MULTIBIAS)
    }},
    {INTEL_CONVOLUTIONAL, {
        ConvolutionalLayer2DCapabilities::GetOperands(BiasOperandIndex).at(INTEL_CONVOLUTIONAL)
    }},
    {INTEL_CONVOLUTIONAL_2D, {
        ConvolutionalLayer2DCapabilities::GetOperands(BiasOperandIndex).at(INTEL_CONVOLUTIONAL_2D)
    }},
    {INTEL_CONVOLUTIONAL_1D, {
        ConvolutionalLayer2DCapabilities::GetOperands(BiasOperandIndex).at(INTEL_CONVOLUTIONAL_1D)
    }},
    {INTEL_GMM, {
        GmmLayerCapabilities::GetOperands(BiasOperandIndex).at(INTEL_GMM)
    }},
    {INTEL_RECURRENT, {
        AffineLayerCapabilities::GetOperands(BiasOperandIndex).at(INTEL_RECURRENT)
    }}
};

const SetLimits<KernelBiasMode> BiasTensor::modeLimits
{
    { KernelBiasModeDisabled, KernelBiasModePerFilter, KernelBiasModePerStride },
    Gna2StatusXnnErrorBiasMode
};

BiasTensor::BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn, Gna2BiasMode biasMode)
try :
    Tensor{ dimensions, dataMode, buffer, Validator{ validatorIn, capabilities } },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, dataMode.Mode) }
{
    validate();
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, BiasOperandIndex);
}

BiasTensor::BiasTensor(const Gna2Tensor &apiTensor, const uint32_t biasVectorIndex,
        Gna2BiasMode biasMode, const LayerValidator& validatorIn)
try :
    Tensor{ apiTensor, capabilities.GetOrder(validatorIn), Validator { validatorIn, capabilities } },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, apiTensor.Mode) }
{
    validate();
}
catch (GnaException& e)
{
    ModelErrorHelper::SetOperandIndexRethrow(e, BiasOperandIndex);
}

void BiasTensor::validate() const
{
    const std::function<void()> command = [&]()
    {
        ModelErrorHelper::ExpectAboveEq(VectorIndex, ui32_0);
        ModelErrorHelper::ExpectBelowEq(VectorIndex, VectorCount - 1);
    };
    ModelErrorHelper::ExecuteForModelItem(command, GNA2_DISABLED, BiasVectorParamIndex);
    Expect::InSet(BiasMode, modeLimits);
}

KernelBiasMode BiasTensor::ToKernelBiasMode(Gna2BiasMode mode, Gna2TensorMode tensorMode)
{
    if (Gna2TensorModeDisabled == tensorMode ||
        Gna2TensorModeConstantScalar == tensorMode)
    {
        return KernelBiasModeDisabled;
    }
    static const std::map<Gna2BiasMode, KernelBiasMode> biasMap
    {
        { Gna2BiasModeDefault, KernelBiasModePerFilter },
        { Gna2BiasModePerStride, KernelBiasModePerStride },
        { Gna2BiasModeGrouping, KernelBiasModePerFilter },
    };
    return biasMap.at(mode);
}
