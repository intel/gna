/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "ConvolutionalLayer2DCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "ParameterLimits.h"
#include "Tensor.h"
#include "Validator.h"

#include <cstdint>
#include <memory.h>
#include <vector>

using namespace GNA;

namespace GNA
{

template<nn_operation operation>
struct ComponentCaps<FilterOperandIndex, operation> : protected LayerCapabilities
{
    static const DataModeLimits& GetModes()
    {
        static const std::map<nn_operation, DataModeLimits> modes =
        {
                {INTEL_CONVOLUTIONAL,
                    {{Gna2DataTypeInt16}, GetError<FilterOperandIndex>().second }},
                {INTEL_CONVOLUTIONAL_2D,
                    {{Gna2DataTypeInt8, Gna2DataTypeInt16}, GetError<FilterOperandIndex>().second }},
                {INTEL_CONVOLUTIONAL_1D,
                    {{Gna2DataTypeInt8, Gna2DataTypeInt16}, GetError<FilterOperandIndex>().second }},
        };
        return modes.at(operation);
    }
};

template<nn_operation operation>
struct ComponentCaps<BiasOperandIndex, operation> : protected LayerCapabilities
{
    static const DataModeLimits& GetModes()
    {
        static const std::map<nn_operation, DataModeLimits> modes =
        {
                {INTEL_CONVOLUTIONAL,
                    { {Gna2DataTypeInt32}, GetError<FilterOperandIndex>().second }},
                {INTEL_CONVOLUTIONAL_2D,
                    { { Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, DataMode{} }, GetError<FilterOperandIndex>().second }},
                {INTEL_CONVOLUTIONAL_1D,
                    { { Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, DataMode{} }, GetError<FilterOperandIndex>().second }},
        };
        return modes.at(operation);
    }
};


template<Gna2DeviceGeneration generation, nn_operation operation, typename ... T>
static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
MakeNHWDInput(T ... dimensions)
{
    return LayerCaps::Make<generation, InputOperandIndex, generation, GNA_TENSOR_NHWD, operation>(std::forward<T>(dimensions)...);
}

template<Gna2DeviceGeneration generation, nn_operation operation, typename ... T>
static std::pair<const Gna2DeviceGeneration, const std::shared_ptr<ComponentLimits>>
MakeNHWDOutput(T ... dimensions)
{
    return LayerCaps::Make<generation, OutputOperandIndex, generation, GNA_TENSOR_NHWD, operation>(std::forward<T>(dimensions)...);
}

template<Gna2DeviceGeneration generation, gna_tensor_order order, nn_operation operation>
static auto MakeFilterCaps(const std::vector<uint32_t>& limits)
{
    return LayerCaps::MakeCaps<generation, order, FilterOperandIndex>(limits,
        ComponentCaps<FilterOperandIndex, operation>::GetModes());
}

template<Gna2DeviceGeneration generation, gna_tensor_order order, nn_operation operation>
static auto MakeBiasCaps(const std::vector<uint32_t>& limits)
{
    return LayerCaps::MakeCaps<generation, order, BiasOperandIndex>(limits,
        ComponentCaps<BiasOperandIndex, operation>::GetModes());
}

const FullCapabilitiesMap & ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                LayerCaps::Make<Gna2DeviceGeneration1_0, InputOperandIndex, Gna2DeviceGeneration0_9, INTEL_CONVOLUTIONAL>(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, MakeLimits<InputEqual1, InputOperandIndex>()},
                    {GNA_DIM_W, MakeLimitsMulti<LegacyInputs, InputOperandIndex>()}}),
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                MakeNHWDInput<Gna2DeviceGeneration3_0, INTEL_CONVOLUTIONAL_2D>(
                    InputEqual1,
                    StaticCaps{16, 384, 1},
                    StaticCaps{16, 240, 1},
                    StaticCaps{8, 384, 8}),
                 MakeNHWDInput<Gna2DeviceGeneration3_1, INTEL_CONVOLUTIONAL_2D>(
                    InputEqual1, Input, Input,
                    StaticCaps{1, Filter2DDepthMax, 1 }),
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                MakeNHWDInput<Gna2DeviceGeneration3_0, INTEL_CONVOLUTIONAL_1D>(
                    InputEqual1, InputEqual1, Input1D, InputEqual1),
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                LayerCaps::Make<Gna2DeviceGeneration1_0, OutputOperandIndex, Gna2DeviceGeneration0_9, GNA_TENSOR_NWD, INTEL_CONVOLUTIONAL>(
                    InputEqual1, Input,
                    StaticCaps{ Filter1DElementsMultiplier, Filter1DCountMax, Filter1DElementsMultiplier}),
            }},
            {INTEL_CONVOLUTIONAL_2D,{
                MakeNHWDOutput<Gna2DeviceGeneration3_0, INTEL_CONVOLUTIONAL_2D>(
                    InputEqual1, Input, Input,
                    StaticCaps{ 1u, Filter2DCountMax, 1u }),
                 MakeNHWDOutput<Gna2DeviceGeneration3_1, INTEL_CONVOLUTIONAL_2D>(
                    InputEqual1, Input, Input,
                    StaticCaps{ 1u, Filter2DCountMax, 1u }),
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                MakeNHWDOutput<Gna2DeviceGeneration3_0, INTEL_CONVOLUTIONAL_1D>(
                    InputEqual1, InputEqual1, Input,
                    StaticCaps{ 1u, Filter2DCountMax, 1u }),
            }},
        }},
        {FilterOperandIndex,{
            {INTEL_CONVOLUTIONAL,{
                MakeFilterCaps<Gna2DeviceGeneration1_0, GNA_TENSOR_NW, INTEL_CONVOLUTIONAL>(
                    // N - # filters, W - # filter coefficients
                    { Filter1DElementsMultiplier, Filter1DCountMax, Filter1DElementsMultiplier,
                     Filter1DElementsMin, Filter1DElementsMax, InputElementCountMultiplier })
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                MakeFilterCaps<Gna2DeviceGeneration3_0, GNA_TENSOR_NHWD, INTEL_CONVOLUTIONAL_2D>(
                    {8, 1024, 8,
                    Filter2DElementsMin, 7, Filter2DElementsMin,
                    Filter2DElementsMin, 7, Filter2DElementsMin, //New limitations for ADL 2D CNN
                    Filter2DElementsMin, Filter2DDepthMax, Filter2DElementsMin}), // Padding to 16B is required for each Kernel
                MakeFilterCaps<Gna2DeviceGeneration3_1, GNA_TENSOR_NHWD, INTEL_CONVOLUTIONAL_2D>(
                    {Filter2DElementsMin, 1024, Filter2DElementsMin,
                    Filter2DElementsMin, Filter2DElementsMax, Filter2DElementsMin,
                    Filter2DElementsMin, Filter2DElementsMax, Filter2DElementsMin,
                    Filter2DElementsMin, Filter2DDepthMax, Filter2DElementsMin}),// Padding to 16B is required for each Kernel
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                LayerCaps::MakeCaps<Gna2DeviceGeneration3_0, GNA_TENSOR_NHWD, FilterOperandIndex>(
                    { Filter1DElementsMultiplier, Filter2DCountMax, Filter1DElementsMultiplier,
                    Filter2DElementsMin, Filter2DElementsMin, Filter2DElementsMin,
                    Filter1DElementsMin, Filter1DElementsMax, Filter1DElementsMin,
                    Filter2DElementsMin, Filter2DElementsMin, Filter2DElementsMin },
                    {Gna2DataTypeInt16}),
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_CONVOLUTIONAL, {
                MakeBiasCaps<Gna2DeviceGeneration1_0, GNA_TENSOR_N, INTEL_CONVOLUTIONAL>(
                    // H - #kernel (GNA_BIAS_PER_KERNEL)
                    {Filter1DElementsMultiplier, Filter1DCountMax, Filter1DElementsMultiplier }),
            }},
            {INTEL_CONVOLUTIONAL_2D, {
                LayerCaps::MakeCaps<Gna2DeviceGeneration3_0, GNA_TENSOR_NHW, BiasOperandIndex>(
                    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1)
                    {1, Filter1DCountMax, 1,
                    1, 1, 1,
                    1, 1, 1},
                    { Gna2DataTypeInt16, Gna2DataTypeInt32, DataMode{} }),
                MakeBiasCaps<Gna2DeviceGeneration3_1, GNA_TENSOR_NHW, INTEL_CONVOLUTIONAL_2D>(
                    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1)
                    {1, Filter2DCountMax, 1,
                    1, 1, 1,
                    1, 1, 1 }),
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                LayerCaps::MakeCaps<Gna2DeviceGeneration3_0, GNA_TENSOR_NHW, BiasOperandIndex>(
                    // N = #kernels + GNA_BIAS_PER_KERNEL (HW=1)
                    {1, Filter1DCountMax, 1,
                    1, 1, 1,
                    1, 1, 1},
                    {Gna2DataTypeInt32}),
            }},
        }},
    };
    return operands.at(operandIndex);
}

const OperationCapabilityMap& ConvolutionalLayer2DCapabilities::GetOperands(uint32_t operandIndex,
    nn_operation operation)
{
    return GetOperands(operandIndex).at(operation);
}

const FullCapabilitiesMap & ConvolutionalLayer2DCapabilities::GetParameters(uint32_t parameterIndex)
{
    static const ComponentFullCapabilityMap parameters =
    {
        {ConvolutionStrideParamIndex,{
            {INTEL_CONVOLUTIONAL_2D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 7, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, 7, 1, Gna2StatusCnnErrorConvFltStride}}}))}, //New limitations for ADL 2D CNN
                { Gna2DeviceGeneration3_1, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}}}))},
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorConvFltStride}},
                    {GNA_DIM_W, {1, Filter1DElementsMax, 1, Gna2StatusCnnErrorConvFltStride}}}))},
            }},
        }},
        {ZeroPaddingParamIndex,{
            {INTEL_CONVOLUTIONAL_2D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                        {{GNA_DIM_H, {0, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltPadding}},
                        {GNA_DIM_W, {0, Filter2DElementsMax, 1, Gna2StatusCnnErrorConvFltPadding}}}))},
            }},
            {INTEL_CONVOLUTIONAL_1D,{
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {0, 0, 1, Gna2StatusCnnErrorConvFltPadding}},
                    {GNA_DIM_W, {0, 0, 1, Gna2StatusCnnErrorConvFltPadding}}}))},
            }},
        }},
        {PoolingStrideParamIndex,{
            {INTEL_CONVOLUTIONAL_2D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 3, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {1, 3, 1, Gna2StatusCnnErrorPoolSize}}}))},
                { Gna2DeviceGeneration3_1, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolStride}},
                    {GNA_DIM_W, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolStride}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorPoolStride}},
                    {GNA_DIM_W, {1, PoolingWindowSizeMax, 1, Gna2StatusCnnErrorPoolStride}}}))},
            }},
        }},
        {PoolingWindowParamIndex,{
            {INTEL_CONVOLUTIONAL_2D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 3, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {1, 3, 1, Gna2StatusCnnErrorPoolSize}}}))},
                { Gna2DeviceGeneration3_1, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {1, Filter2DElementsMax, 1, Gna2StatusCnnErrorPoolSize}}}))}
            }},
            {INTEL_CONVOLUTIONAL_1D, {
                { Gna2DeviceGeneration3_0, std::make_shared<ComponentLimits>(ComponentLimits(
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, 1, 1, Gna2StatusCnnErrorPoolSize}},
                    {GNA_DIM_W, {0, PoolingWindowSizeMax, 1, Gna2StatusCnnErrorPoolSize}}}))},
            }},
        }},
    };
    return parameters.at(parameterIndex);
}

const OperationCapabilityMap& ConvolutionalLayer2DCapabilities::GetParameters(uint32_t parameterIndex,
    nn_operation operation)
{
    return GetParameters(parameterIndex).at(operation);
}

}
