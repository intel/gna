/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "TransposeLayer.h"

#include "AccelerationDetector.h"
#include "Expect.h"
#include "LayerConfiguration.h"
#include "Macros.h"

using namespace GNA;

TransposeLayer::TransposeLayer(
        const Gna2Operation& operation,
        const LayerValidator& validatorIn) :
    Layer(operation, validatorIn, {}, BaseAddress{}),
    transposeKernels{ AccelerationDetector::GetKernelMap<TransposeKernel>(
                                            KERNEL_TRANSPOSE, KernelMode{Input.Mode}) }
{
    auto *inputTensor = reinterpret_cast<const Gna2Tensor *>(operation.Operands[InputOperandIndex]);
    Expect::Equal(inputTensor->Shape.NumberOfDimensions,
            static_cast<uint32_t>(2), Gna2StatusXnnErrorLyrCfg);

    auto *outputTensor = reinterpret_cast<const Gna2Tensor *>(operation.Operands[OutputOperandIndex]);
    Expect::Equal(outputTensor->Shape.NumberOfDimensions,
            static_cast<uint32_t>(2), Gna2StatusXnnErrorLyrCfg);

    transposeHiddenConfig = std::make_unique<TransposeConfig>(
            Input.Dimensions.at('H'), Input.Dimensions.at('W'), Input.Buffer, Output.Buffer);

    ModelErrorHelper::ExpectEqual(Output.at('H'), Input.at('W'));
    ModelErrorHelper::ExpectEqual(Output.at('W'), Input.at('H'));

    Expect::Null(Output.ScratchPad); // in transpose layer no 4B output array is allowed

    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
                    {this->compute(layerConfiguration, accel, executionConfig); };

    dataConfig = { Input.Mode, DataMode{}, DataMode{}, Output.Mode };
}

void TransposeLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputOperandIndex) != 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputOperandIndex];
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputOperandIndex) != 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputOperandIndex];
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Transpose)
    {
        configs.Transpose = std::make_unique<TransposeConfig>(*transposeHiddenConfig);
    }

    configs.Transpose->input = inputBuffer;
    configs.Transpose->output = outputBuffer;
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
