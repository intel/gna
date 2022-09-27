/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#define NOMINMAX 1

#include "Bias.h"

#include "AffineLayerCapabilities.h"
#include "Capabilities.h"
#include "ConvolutionKernelArguments.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "GmmLayerCapabilities.h"
#include "ModelError.h"
#include "ParameterLimits.h"
#include "PoolingFunctions2D.h"
#include "Shape.h"
#include "Validator.h"

#include "gna2-common-api.h"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <utility>

using namespace GNA;

const FullCapabilitiesMap BiasTensor::capabilities = LayerCapabilities::MakeFullCaps<BiasOperandIndex>();

const SetLimits<KernelBiasMode> BiasTensor::modeLimits
{
    { KernelBiasModeDisabled, KernelBiasModePerFilter, KernelBiasModePerStride },
    Gna2StatusXnnErrorBiasMode
};

BiasTensor::BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
    void * buffer, const LayerValidator& validatorIn, Gna2BiasMode biasMode)
try :
    Tensor{ dimensions, dataMode, buffer, Validator{ validatorIn, capabilities }, BiasOperandIndex },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, dataMode.Mode) }
{
    validate();
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(BiasOperandIndex);
}

BiasTensor::BiasTensor(const Gna2Tensor &apiTensor, const uint32_t biasVectorIndex,
        Gna2BiasMode biasMode, const LayerValidator& validatorIn)
try :
    Tensor{ apiTensor, capabilities.GetOrder(validatorIn), Validator { validatorIn, capabilities }, BiasOperandIndex },
    VectorCount{ biasMode == Gna2BiasModeGrouping ? Dimensions.at('W') : 1 },
    VectorIndex{ biasVectorIndex },
    BiasMode{ ToKernelBiasMode(biasMode, apiTensor.Mode) }
{
    validate();
}
catch (GnaException&)
{
    GnaModelErrorException::DispatchAndFill(BiasOperandIndex);
}

void BiasTensor::validate() const
{
    auto const ctx = ModelItem{ Gna2ItemTypeParameter, Gna2DisabledU32, BiasVectorParamIndex };
    ModelErrorHelper::ExpectAboveEq(VectorIndex, 0u, ctx);
    ModelErrorHelper::ExpectBelowEq(VectorIndex, VectorCount - 1, ctx);

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
