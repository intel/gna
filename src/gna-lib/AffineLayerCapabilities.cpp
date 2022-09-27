/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "AffineLayerCapabilities.h"

#include "DataMode.h"
#include "Capabilities.h"
#include "Tensor.h"

#include "LayerCapabilities.h"

using namespace GNA;

namespace GNA
{

template<>
struct ComponentCaps<BiasOperandIndex, INTEL_AFFINE> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static auto Make()
    {
        return LayerCaps::Make<generation, BiasOperandIndex, GNA_TENSOR_H, INTEL_AFFINE>(
            Input);
    }

    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        return GetCommonModes(BiasOperandIndex, generation);
    }
};

template<>
struct ComponentCaps<BiasOperandIndex, INTEL_AFFINE_DIAGONAL> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static auto Make()
    {
        return ComponentCaps<BiasOperandIndex, INTEL_AFFINE>::Make<generation>();
    }
};

template<nn_operation operation>
struct ComponentCaps<WeightOperandIndex, operation> : protected LayerCapabilities
{
    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        static const std::map<Gna2DeviceGeneration, DataModeLimits> modes =
        {
            MakeModes<Gna2DeviceGeneration0_9, WeightOperandIndex>
                (Gna2DataTypeInt8, Gna2DataTypeInt16),
            MakeModes<Gna2DeviceGeneration2_0, WeightOperandIndex>
                (Gna2DataTypeInt8, Gna2DataTypeInt16),
        };
        return modes.at(generation);
    }
};

template<>
struct ComponentCaps<WeightScaleFactorOperandIndex, INTEL_AFFINE_MULTIBIAS> : protected LayerCapabilities
{
    static const DataModeLimits& GetModes(Gna2DeviceGeneration generation)
    {
        static const std::map<Gna2DeviceGeneration, DataModeLimits> modes =
        {
            MakeModes<Gna2DeviceGeneration2_0, WeightScaleFactorOperandIndex>
                (Gna2DataTypeWeightScaleFactor),
        };
        return modes.at(generation);
    }
};

template<>
struct ComponentCaps<InputOperandIndex, INTEL_RECURRENT> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, MakeLimits<InputGroupMax, InputOperandIndex>()},
                    {GNA_DIM_W, MakeLimitsMulti<LegacyInputs, InputOperandIndex>()}},
                GetCommonModes(OutputOperandIndex, modeGeneration)}) };
    }
};

template<>
struct ComponentCaps<OutputOperandIndex, INTEL_RECURRENT> : protected LayerCapabilities
{
    template<Gna2DeviceGeneration generation, Gna2DeviceGeneration modeGeneration = generation>
    static std::pair<const Gna2DeviceGeneration, std::shared_ptr<ComponentLimits>>
    Make()
    {
        return { generation,
            std::make_shared<TensorLimits>(TensorLimits{
                {GNA_TENSOR_HW},
                {{GNA_DIM_H, MakeLimits<InputGroupMax, OutputOperandIndex>()},
                    {GNA_DIM_W, MakeLimits<OutputRnn, OutputOperandIndex>()}},
                // must be multiple 32 to keep 64B output buffer alignment
                GetCommonModes(OutputOperandIndex, modeGeneration)}) };
    }
};

const FullCapabilitiesMap& AffineLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            LayerCaps::MakeAllGensSame<InputOperandIndex, INTEL_AFFINE>(),
            LayerCaps::MakeAllGensSame<InputOperandIndex, INTEL_AFFINE_DIAGONAL>(),
            LayerCaps::MakeAllGensSame<InputOperandIndex, INTEL_AFFINE_MULTIBIAS>(),
            LayerCaps::MakeAllGensSame<InputOperandIndex, INTEL_RECURRENT>(),
        }},
        {OutputOperandIndex,{
            LayerCaps::MakeAllGensSame<OutputOperandIndex, INTEL_AFFINE>(),
            LayerCaps::MakeAllGensSame<OutputOperandIndex, INTEL_AFFINE_DIAGONAL>(),
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0, OutputOperandIndex, GNA_TENSOR_HW, INTEL_AFFINE_MULTIBIAS>(
            StaticCaps{ Input[0], Input[1], 8 }, InputGroupMax),
                LayerCaps::Make<Gna2DeviceGeneration3_0, OutputOperandIndex, INTEL_AFFINE_MULTIBIAS>(),
            }},

            LayerCaps::MakeAllGensSame<OutputOperandIndex, INTEL_RECURRENT>(),
        }},
        {WeightOperandIndex,{
            {INTEL_AFFINE, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, WeightOperandIndex, GNA_TENSOR_HW, INTEL_AFFINE>(
                    Input, WeightMultiplier),
            }},
            {INTEL_AFFINE_DIAGONAL, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, WeightOperandIndex, GNA_TENSOR_H, INTEL_AFFINE_DIAGONAL>(
                    WeightMultiplier)
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0, WeightOperandIndex, GNA_TENSOR_HW, INTEL_AFFINE_MULTIBIAS>(
                    // W - #inputs, H - #outputs
                    Input, WeightMultiplier)
            }},
            {INTEL_RECURRENT, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, WeightOperandIndex, GNA_TENSOR_HW, INTEL_RECURRENT>(
                    OutputRnn, StaticCaps{InputElementCountMultiplier + RecurrentOutputElementCountMultiplier,
                                InputElementCountMax + InputElementCountMax,
                                InputElementCountMultiplier})
            }},
        }},
        {BiasOperandIndex,{
            LayerCaps::MakeAllGensSame<BiasOperandIndex, INTEL_AFFINE>(),
            LayerCaps::MakeAllGensSame<BiasOperandIndex, INTEL_AFFINE_DIAGONAL>(),
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0>(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, MakeLimits<Input, BiasOperandIndex>()},
                    {GNA_DIM_W, MakeLimits<InputGroupMax, BiasOperandIndex>()}},
                    {{Gna2DataTypeInt32 }, Gna2StatusXnnErrorBiasBytes }),
                LayerCaps::Make<Gna2DeviceGeneration3_0>(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, MakeLimits<Input, BiasOperandIndex>()},
                    {GNA_DIM_W, MakeLimits<InputGroupMax, BiasOperandIndex>()}},
                    {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32},
                            Gna2StatusXnnErrorBiasBytes}),
            }},
            {INTEL_RECURRENT, {
                LayerCaps::Make<Gna2DeviceGeneration0_9, BiasOperandIndex, GNA_TENSOR_H, INTEL_RECURRENT>(
                    OutputRnn),
                LayerCaps::Make<Gna2DeviceGeneration3_0, BiasOperandIndex, GNA_TENSOR_H, INTEL_RECURRENT>(
                    OutputRnn),
            }},
        }},
        { WeightScaleFactorOperandIndex,{
            {INTEL_AFFINE_MULTIBIAS, {
                LayerCaps::Make<Gna2DeviceGeneration2_0, WeightScaleFactorOperandIndex, GNA_TENSOR_H, INTEL_AFFINE_MULTIBIAS>(
                    Input),
        }},
    }}
    };

    return operands.at(operandIndex);
}

}