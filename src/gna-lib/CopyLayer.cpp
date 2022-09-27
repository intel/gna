/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "CopyLayer.h"

#include "AccelerationDetector.h"
#include "Address.h"
#include "Capabilities.h"
#include "Expect.h"
#include "LayerCapabilities.h"
#include "LayerConfiguration.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Macros.h"
#include "ParameterLimits.h"

#include "gna2-common-api.h"
#include "gna2-model-api.h"


#include <algorithm>
#include <memory>

namespace GNA
{
class BaseValidator;
}

using namespace GNA;

static const std::pair<gna_tensor_dim, RangeLimits<uint32_t>> copyShapeWlimit =
{ GNA_DIM_W, RangeLimits<uint32_t>{LayerCapabilities::LegacyInputs, Gna2StatusXnnErrorLyrCfg } };

const FullCapabilitiesMap CopyLayer::limits
{
    { INTEL_COPY, {
        { Gna2DeviceGeneration0_9, std::make_shared<ComponentLimits>(ComponentLimits(
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, {1, BatchSizeMax, 1, Gna2StatusXnnErrorLyrCfg}},
            copyShapeWlimit}))},
    }},
};

CopyLayer::CopyLayer(const Gna2Operation& operation, const LayerValidator& validatorIn) :
    Layer(operation, validatorIn, {}, BaseAddress()),
    ColumnCount{ GetCopyShape(operation).at('W') },
    RowCount{ GetCopyShape(operation).at('H') },
    copyKernels{ AccelerationDetector::GetKernelMap<CopyKernel>(KERNEL_COPY, KernelMode {Input.Mode}) },
    copyHiddenConfig{ RowCount, ColumnCount, Input.Dimensions.at('W'), Output.Dimensions.at('W'), Input.Buffer, Output.Buffer }
{

    try
    {
        auto const copyParams = std::make_unique<const Component>(Shape{ GNA_TENSOR_HW, RowCount, ColumnCount },
            Validator{ *validator, limits }, false, CopyShapeParamIndex);
        copyParams->ValidateDimensions(Input.Mode.Type);
        ModelErrorHelper::ExpectBelowEq(RowCount, Input.Dimensions.at('H'), Gna2ItemTypeShapeDimensions);
    }
    catch(GnaModelErrorException&)
    {
        GnaModelErrorException::DispatchAndFill(Gna2DisabledU32, 0);
    }
    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->computeHidden(accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(layerConfiguration, accel, executionConfig); };
    dataConfig = { Input.Mode, DataMode{}, DataMode{}, Output.Mode };
}

void CopyLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    BaseAddress inputBuffer = Input;
    if (layerConfiguration.Buffers.count(InputOperandIndex) > 0)
    {
        inputBuffer = layerConfiguration.Buffers[InputOperandIndex];
    }

    BaseAddress outputBuffer = Output;
    if (layerConfiguration.Buffers.count(OutputOperandIndex) > 0)
    {
        outputBuffer = layerConfiguration.Buffers[OutputOperandIndex];
    }

    auto& configs = layerConfiguration.Configs;
    if(!configs.Copy)
    {
        configs.Copy = std::make_unique<CopyConfig>(copyHiddenConfig);
    }

    configs.Copy->input = inputBuffer;
    configs.Copy->output = outputBuffer;
}

void CopyLayer::computeHidden(AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    copyKernels.at(accel)(&copyHiddenConfig);
}

void CopyLayer::compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig) const
{
    UNREFERENCED_PARAMETER(executionConfig);
    auto copyConfig = layerConfiguration.Configs.Copy.get();
    copyKernels.at(accel)(copyConfig);
}

Shape CopyLayer::GetCopyShape(const Gna2Operation& operation)
{
    return Shape::Create(*reinterpret_cast<const Gna2Shape *>(operation.Parameters[0]), GNA_TENSOR_HW);
}
