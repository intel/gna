/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ConvolutionalLayer2D.h"

#include "ActivationFunction.h"
#include "Address.h"
#include "ConvolutionalFunctions2D.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "Expect.h"
#include "HardwareCapabilities.h"
#include "HardwareLayer.h"
#include "KernelArguments.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "PoolingFunctions2D.h"
#include "Tensor.h"
#include "Transform.h"
#include "TransformMap.h"
#include "Validator.h"

#include "gna-api-types-xnn.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace GNA;

void ConvolutionalLayer2D::Init()
{
    if (inputTransform->Is1D() &&
        (Transforms.Get<PoolingFunction2D>(PoolingTransform2D) == nullptr ||outputTransform->Is1D()))
    {
        auto const & capsMapIn = ConvolutionalLayer2DCapabilities::GetOperands(InputOperandIndex);
        Input.Validate(capsMapIn, INTEL_CONVOLUTIONAL_1D);

        auto const & capsMapOut = ConvolutionalLayer2DCapabilities::GetOperands(OutputOperandIndex);
        Output.Validate(capsMapOut, INTEL_CONVOLUTIONAL_1D);
    }
    else
    {
        auto const precision = Output.Mode.Size;
        if (precision < 4)
        {
            auto const & filters = getTransformOperand(ConvolutionalTransform2D, FilterOperandIndex);
            auto const filterCount = filters.at(GNA_DIM_N);
            if (filterCount > 2)
            {
                Expect::MultiplicityOf(filterCount, 4 / precision,
                    Gna2StatusCnnErrorConvFltCount);
            }
        }
    }

    Expect::One(Input.at(GNA_DIM_N), Gna2StatusXnnErrorGrouping);
    Expect::One(Output.at(GNA_DIM_N), Gna2StatusXnnErrorGrouping);
    Expect::Equal(Output.Size, GetOutputTransform()->Output->Size, Gna2StatusXnnErrorOutputVolume);
}

Tensor const & ConvolutionalLayer2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        if (Transforms.Get(ActivationTransform))
        {
            return Output.ScratchPad;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case FilterOperandIndex://[[fallthrough]]
    case BiasOperandIndex:
    {
        return getTransformOperand(ConvolutionalTransform2D, operandIndex);
    }
    case PwlOperandIndex:
    {
        return getTransformOperand(ActivationTransform, 2);
    }
    case SoftwareScratchpadOperandIndex:
    {
        if (Transforms.size() > 1)
        {
            return inputTransform->GetOperand(OutputOperandIndex);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    default:
        return Layer::GetOperand(operandIndex);
    }
}

DataConfig ConvolutionalLayer2D::GetDataMode() const
{
    auto& convolutionTransform = *Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D);
    const auto filterMode = convolutionTransform.Filters->Mode.Value;
    const auto biasMode = convolutionTransform.Biases->Mode.Value;
    return DataConfig(Input.Mode.Value, filterMode, biasMode, Output.Mode.Value);
}
