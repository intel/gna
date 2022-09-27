/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "AuxiliaryCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "Tensor.h"


using namespace GNA;

namespace GNA
{

template<uint32_t operandIndex>
struct AuxComponentCaps : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
        Make()
    {
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, MakeLimits<InputGroupMax, operandIndex>()},
                    {GNA_DIM_W, MakeLimitsMulti<LegacyInputs, operandIndex>()}},
                GetCommonModes(operandIndex, modeGeneration)}) };
    }

    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
        MakeInterleaved()
    {
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, MakeLimitsMulti<LegacyInputs, operandIndex>()},
                    {GNA_DIM_W, MakeLimits<InputGroupMax, operandIndex>()}},
                GetCommonModes(operandIndex, modeGeneration)}) };
    }
};

using InputAuxCaps = AuxComponentCaps<InputOperandIndex>;
using OutputAuxCaps = AuxComponentCaps<OutputOperandIndex>;

const FullCapabilitiesMap& AuxiliaryCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_COPY, {
                InputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                InputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
            }},
            {INTEL_INTERLEAVE, {
                InputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                InputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
            }},
            {INTEL_DEINTERLEAVE, {
                InputAuxCaps::MakeInterleaved<Gna2DeviceGeneration0_9>(),
                InputAuxCaps::MakeInterleaved<Gna2DeviceGeneration3_0>(),
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_COPY, {
                OutputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                OutputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
            }},
            {INTEL_DEINTERLEAVE, {
                OutputAuxCaps::Make<Gna2DeviceGeneration0_9>(),
                OutputAuxCaps::Make<Gna2DeviceGeneration3_0>(),
            }},
            {INTEL_INTERLEAVE, {
                OutputAuxCaps::MakeInterleaved<Gna2DeviceGeneration0_9>(),
                OutputAuxCaps::MakeInterleaved<Gna2DeviceGeneration3_0>(),
            }},
        }},
    };

    return operands.at(operandIndex);
}

}
