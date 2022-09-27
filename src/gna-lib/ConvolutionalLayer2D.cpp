/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
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

#include <algorithm>
#include <memory>
#include <stdexcept>

using namespace GNA;

void ConvolutionalLayer2D::Init()
{
    Expect::Null(Output.ScratchPad);
    if(Operation == INTEL_CONVOLUTIONAL_2D)
    {
        Validate3_0ExtraLimits();
    }
    const std::function<void()> command = [&]()
    {
        Output.Dimensions.ExpectEqual(GetOutputTransform().Output->Dimensions);
    };
    ModelErrorHelper::ExecuteForModelItem(command, OutputOperandIndex);
    ModelErrorHelper::ExpectEqual(Output.Mode.Type, GetOutputTransform().Output->Mode.Type, ModelItem{ Gna2ItemTypeOperandType, OutputOperandIndex });
    ModelErrorHelper::ExpectEqual(Output.Size, GetOutputTransform().Output->Size, ModelItem{ Gna2ItemTypeOperandMode, OutputOperandIndex });

    auto const & convolutionTransform = Transforms.Get<ConvolutionFunction2D>(ConvolutionalTransform2D);
    const auto filterMode = convolutionTransform.Filters->Mode;
    const auto biasMode = convolutionTransform.Biases->Mode;
    auto const activation = Transforms.GetOptional<ActivationFunction>(ActivationTransform);
    dataConfig = { Input.Mode, filterMode, biasMode, Output.Mode, activation == nullptr };
}

void ConvolutionalLayer2D::Validate3_0ExtraLimits() const
{
    if(validator->Generation <= Gna2DeviceGeneration3_1)
    {
        auto const precision = Output.Mode.Size;
        if (precision < 4)
        {
            auto const& filters = getTransformOperand(ConvolutionalTransform2D, FilterOperandIndex);
            auto const filterCount = filters.at(GNA_DIM_N);
            if (filterCount > 2)
            {
                ModelErrorHelper::ExpectMultiplicityOf(filters.at('N'), 4 / precision);
            }
        }
    }

    if (Gna2DeviceGeneration3_0 == validator->Generation)
    {
        auto const activation = Transforms.GetOptional(ActivationTransform);
        auto const pooling = Transforms.GetOptional<PoolingFunction2D>(PoolingTransform2D);
        auto const & filter = GetInputTransform().GetOperand(FilterOperandIndex);
        ModelErrorHelper::ExpectEqual(filter.Mode.Mode, Input.Mode.Mode, ModelItem{ Gna2ItemTypeOperandMode, FilterOperandIndex });
        ModelErrorHelper::ExpectEqual(filter.Mode.Type, Input.Mode.Type, ModelItem{ Gna2ItemTypeOperandType, FilterOperandIndex });
        if (activation)
        {
            ModelErrorHelper::ExpectEqual(Output.Mode.Mode, Input.Mode.Mode, ModelItem{ Gna2ItemTypeOperandMode, OutputOperandIndex, });
            ModelErrorHelper::ExpectEqual(Output.Mode.Type, Input.Mode.Type, ModelItem{ Gna2ItemTypeOperandType, OutputOperandIndex, });
        }
        if (pooling)
        {
            auto const command = [&]()
            {
                pooling->Window->Dimensions.ExpectSquare();
            };
            ModelErrorHelper::ExecuteForModelItem(command, Gna2DisabledU32, PoolingWindowParamIndex);
        }
        const auto kDim = filter.Dimensions.Reshape(GNA_TENSOR_HW);
        if (Gna2DataTypeInt16 == filter.Mode.Type)
        {
            if (filter.at(GNA_DIM_W) > 1)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 120u);
            }
            if (filter.at(GNA_DIM_W) > 3)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 80u);
            }
            if (filter.at(GNA_DIM_W) > 4)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 64u);
            }
            if (filter.at(GNA_DIM_W) > 5)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 48u);
            }
        }
        else
        {
            if (filter.at(GNA_DIM_W) > 2)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 240u);
            }
            if (filter.at(GNA_DIM_W) > 3)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 168u);
            }
            if (filter.at(GNA_DIM_W) > 4)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 136u);
            }
            if (filter.at(GNA_DIM_W) > 5)
            {
                ModelErrorHelper::ExpectBelowEq(Input.at('D'), 96u);
            }
        }
    }
}

Tensor const & ConvolutionalLayer2D::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case FilterOperandIndex:
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
            return GetInputTransform().GetOperand(OutputOperandIndex);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
    default:
        return Layer::GetOperand(operandIndex);
    }
}

std::unique_ptr<const Component> ConvolutionalLayer2D::CreateComponentFromParameter(const Shape& shape,
    const LayerValidator& validatorIn, const uint32_t parameterIndex)
{
    auto const command = [&]()
    {
        return OperationConfig::CreateCnnComponent(shape,
            validatorIn, ConvolutionalLayer2DCapabilities::GetParameters(parameterIndex), parameterIndex);
    };
    return ModelErrorHelper::ExecuteForModelItem(command, Gna2DisabledU32, parameterIndex);
}
