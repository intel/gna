/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ConvolutionalLayer.h"

#include "Address.h"
#include "DataMode.h"
#include "Expect.h"
#include "KernelArguments.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Macros.h"
#include "Shape.h"
#include "Tensor.h"


#include <algorithm>
#include <cstdint>
#include <map>

using namespace GNA;

CnnLayer::CnnLayer(const Gna2Operation& apiLayer, const LayerValidator& validatorIn) :
    Layer(apiLayer, validatorIn, {}, BaseAddress())
{
    ExpectValid();
    Convolution = GetConvolution(getDetails(apiLayer));
    Activation = ActivationFunction::Create({ &Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
        apiLayer, *validator });
    Pooling = GetPooling(apiLayer);
    Init();
    dataConfig = DataConfig{ Input.Mode, Convolution->Filters->Mode,
        Convolution->Biases->Mode, Output.Mode, Activation == nullptr };
}

void CnnLayer::ExpectValid() const
{
    Expect::Equal(validator->Operation, INTEL_CONVOLUTIONAL, Gna2StatusXnnErrorLyrOperation);
    Expect::One(Input.Grouping, Gna2StatusXnnErrorGrouping);
    Expect::One(Output.Grouping, Gna2StatusXnnErrorGrouping);
}

std::unique_ptr<const PoolingFunction> CnnLayer::GetPooling(const Gna2Operation& apiOperation) const
{
    return PoolingFunction::Create(apiOperation, Convolution->Output, *validator, Input.Mode);
}

void CnnLayer::Init()
{
    uint32_t outputsPerFilter = Convolution->OutputsPerFilterCount;
    auto effectiveComputeHidden =  &CnnLayer::computeHidden;
    auto effectiveCompute = &CnnLayer::compute;
    if (Pooling)
    {
        outputsPerFilter = Pooling->OutputsPerFilterCount;
        // Activation is required for cnn with pooling
        ModelErrorHelper::ExpectNotNull(Activation.get(), Gna2ItemTypeOperationOperands, PwlOperandIndex);
        effectiveComputeHidden = &CnnLayer::computeHiddenPool;
        effectiveCompute = &CnnLayer::computePool;
    }
    else if (Activation)
    {
        effectiveComputeHidden = &CnnLayer::computeHiddenPwl;
        effectiveCompute = &CnnLayer::computePwl;
    }
    ModelErrorHelper::ExpectEqual(Output.at('W'), outputsPerFilter);

    Layer::ComputeHidden = [this, effectiveComputeHidden](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {(this->*effectiveComputeHidden)(accel, executionConfig); };

    Layer::Compute = [this, effectiveCompute](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
    {(this->*effectiveCompute)(layerConfiguration, accel, executionConfig); };
}

Tensor const & CnnLayer::GetOperand(uint32_t operandIndex) const
{

    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        if (Activation)
        {
            return Output.ScratchPad;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case FilterOperandIndex:
        if (Convolution)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Convolution->Filters);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case BiasOperandIndex:
        if (Convolution)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Convolution->Biases);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case PwlOperandIndex:
        if (Activation)
        {
            return BaseTransform::GetOperandIfExistOrThrow(Activation->Segments);
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    default:
        return Layer::GetOperand(operandIndex);
    }
}

void CnnLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    Layer::UpdateKernelConfigs(layerConfiguration);

    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputOperandIndex) > 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputOperandIndex];
    }

    BaseAddress filterOutputBuffer = Activation ? Output.ScratchPad :
        (layerConfiguration.Buffers.count(OutputOperandIndex) > 0 ? layerConfiguration.Buffers[OutputOperandIndex] : Output);

    BaseAddress pwlOutputBuffer = layerConfiguration.Buffers.count(OutputOperandIndex) > 0
        ? layerConfiguration.Buffers[OutputOperandIndex]
        : Output;

    if (layerConfiguration.Buffers.count(OutputOperandIndex) > 0)
    {
        if (Activation)
        {
            Output.ValidateBuffer(pwlOutputBuffer);
        }
        else
        {
            Output.ValidateBuffer(filterOutputBuffer);
        }
    }

    auto& configs = layerConfiguration.Configs;
    if (!Pooling)
    {
        configs.Convolution = Convolution->GetRequestConfig(inputBuffer, filterOutputBuffer);
        if (Activation)
        {
            Activation->UpdateConfigBuffers(layerConfiguration.ConfigList,
                { Output.ScratchPad, pwlOutputBuffer });
        }
    }
    else
    {
        configs.Convolution = Convolution->GetRequestConfig(inputBuffer, pwlOutputBuffer);
    }
}

bool CnnLayer::IsForced(const Gna2Operation& operation)
{
    return strncmp(operation.Operands[OutputOperandIndex]->Layout, "GNA1", 4) == 0;
}

void CnnLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->ComputeHidden(accel, execution);
}

void CnnLayer::computeHiddenPwl(AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->ComputeHidden(accel, execution);
    Activation->Compute(accel, nullptr, execution);
}

void CnnLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->Compute(layerConfiguration.Configs.Convolution.get(), accel, execution);
}

void CnnLayer::computePwl(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    UNREFERENCED_PARAMETER(execution.Intermediate);

    Convolution->Compute(layerConfiguration.Configs.Convolution.get(), accel, execution);
    Activation->Compute(accel, &layerConfiguration, execution);
}

void CnnLayer::computeHiddenPool(AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ Convolution->GetHiddenConfig(), execution };

    convConfig.pooledOutputs = Output.Buffer;

    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, Activation->Pwl.get());
}

void CnnLayer::computePool(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ layerConfiguration.Configs.Convolution.get(), execution };
    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, Activation->Pwl.get());
}

const Gna2Operation & CnnLayer::getDetails(const Gna2Operation & operation)
{
    return operation;
}
