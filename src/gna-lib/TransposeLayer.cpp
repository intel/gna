/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "TransposeLayer.h"

#include "AccelerationDetector.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "Macros.h"

using namespace GNA;

TransposeLayer::TransposeLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    Layer(layer, validatorIn, {}, BaseAddress()),
    transposeKernels{ AccelerationDetector::GetKernelMap<TransposeKernel>(
                                            KERNEL_TRANSPOSE,  KernelMode{Input.Mode}) },
    transposeHiddenConfig(std::make_unique<TransposeConfig>(
                Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, Output.Buffer))
{
    Expect::Equal(Input.Dimensions.at('W'), Output.Dimensions.at('H'), Gna2StatusXnnErrorLyrCfg);
    Expect::Equal(Input.Dimensions.at('H'), Output.Dimensions.at('W'), Gna2StatusXnnErrorLyrCfg);
    Expect::Null(layer.pLayerStruct); // transpose layers do not have layer details

    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

TransposeLayer::TransposeLayer(
        const Gna2Operation& apiOperation,
        const BaseValidator& validatorIn) :
    Layer(apiOperation, validatorIn, {}, BaseAddress{}),
    transposeKernels{ AccelerationDetector::GetKernelMap<TransposeKernel>(
                                            KERNEL_TRANSPOSE, KernelMode{Input.Mode}) }
{
    auto *inputTensor = reinterpret_cast<const Gna2Tensor *>(apiOperation.Operands[InputOperandIndex]);
    Expect::Equal(inputTensor->Shape.NumberOfDimensions,
            static_cast<uint32_t>(2), Gna2StatusXnnErrorLyrCfg);

    auto *outputTensor = reinterpret_cast<const Gna2Tensor *>(apiOperation.Operands[OutputOperandIndex]);
    Expect::Equal(outputTensor->Shape.NumberOfDimensions,
            static_cast<uint32_t>(2), Gna2StatusXnnErrorLyrCfg);

    transposeHiddenConfig = std::make_unique<TransposeConfig>(
            Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, Output.Buffer);

    ModelErrorHelper::ExpectEqual(Output.AsModelValue('H'), Input.AsModelValue('W'));
    ModelErrorHelper::ExpectEqual(Output.AsModelValue('W'), Input.AsModelValue('H'));

    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };
}

void TransposeLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputOperandIndex) != 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputOperandIndex];
        Input.ValidateBuffer(inputBuffer);
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputOperandIndex) != 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputOperandIndex];
        Output.ValidateBuffer(outputBuffer);
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Transpose)
    {
        configs.Transpose = std::make_unique<TransposeConfig>(*transposeHiddenConfig);
    }

    configs.Transpose->input = inputBuffer;
    configs.Transpose->output = outputBuffer;
}

DataConfig TransposeLayer::GetDataMode() const
{
    return DataConfig(Input.Mode, GNA_DATA_DISABLED, GNA_DATA_DISABLED, Output.Mode);
}

void TransposeLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    transposeKernels.at(accel)(transposeHiddenConfig.get());
}

void TransposeLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    auto transposeConfig = layerConfiguration.Configs.Transpose.get();
    transposeKernels.at(accel)(transposeConfig);
}
