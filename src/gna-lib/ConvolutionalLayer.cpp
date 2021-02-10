/**
 @copyright (C) 2018-2021 Intel Corporation
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

#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <algorithm>
#include <cstdint>
#include <map>

using namespace GNA;

void CnnLayer::ExpectValid() const
{
    Expect::Equal(validator->Operation, INTEL_CONVOLUTIONAL, Gna2StatusXnnErrorLyrOperation);
    Expect::One(Input.Grouping, Gna2StatusXnnErrorGrouping);
    Expect::One(Output.Grouping, Gna2StatusXnnErrorGrouping);
}

std::unique_ptr<const PoolingFunction> CnnLayer::GetPooling(const nn_layer& layer) const
{
    return PoolingFunction::Create(layer.pLayerStruct, Convolution->Output, *validator, Input.Mode);
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
    const auto& declaredOutputPerFilter = Output.AsModelValue('W').SetOperand(OutputOperandIndex);
    ModelErrorHelper::ExpectEqual(declaredOutputPerFilter, outputsPerFilter);

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
        Input.ValidateBuffer(inputBuffer);
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
const char * enforcingOutputTensorLayout = "GNA1";
std::unique_ptr<GNA::Layer> CnnLayer::CreateEnforced(const Gna2Operation& operation,
    const BaseValidator& validatorIn)
{
    auto & outputTensor = *const_cast<Gna2Tensor*>(operation.Operands[OutputOperandIndex]);
    const auto outputTensorCopy = outputTensor;
    ModelWrapper::SetLayout(outputTensor, enforcingOutputTensorLayout);
    auto enforcedLayer = std::make_unique<CnnLayer>(operation, validatorIn);
    outputTensor = outputTensorCopy;
    return enforcedLayer;
}

bool CnnLayer::IsForced(const Gna2Operation& operation)
{
    // For the 2.0 releases (matching GNA HW up to 2.0) CNN is always dispatched as Legacy CNN
    return operation.Type == Gna2OperationTypeConvolution;
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

    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}

void CnnLayer::computePool(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const
{
    auto convConfig = ConvolutionConfig{ layerConfiguration.Configs.Convolution.get(), execution };
    Pooling->Compute(&convConfig, accel, execution.Intermediate->pool, &Activation->Pwl);
}

DataConfig CnnLayer::GetDataMode() const
{
    return DataConfig(Input.Mode.Value, Convolution->Filters->Mode.Value,
        Convolution->Biases->Mode.Value, Output.Mode.Value);
}

const nn_layer_conv & CnnLayer::getDetails(const nn_layer & cnn1DLayer)
{
    Expect::NotNull(cnn1DLayer.pLayerStruct);
    return *reinterpret_cast<const nn_layer_conv*>(cnn1DLayer.pLayerStruct);
}

const Gna2Operation & CnnLayer::getDetails(const Gna2Operation & operation)
{
    return operation;
}
