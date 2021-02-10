/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "AffineLayerCapabilities.h"

#include "DataMode.h"
#include "Capabilities.h"
#include "Tensor.h"

#include "LayerCapabilities.h"

using namespace GNA;

static const DataModeLimits& _ModesWeightGen0_9()
{
    static const DataModeLimits __ModesWeightGen0_9 =
    {
        {GNA_INT8, GNA_INT16},
        Gna2StatusXnnErrorWeightBytes
    };
    return __ModesWeightGen0_9;
}

static const DataModeLimits& _ModesBiasGen0_9()
{
    static const DataModeLimits __ModesBiasGen0_9 =
    {
        {GNA_INT32, GNA_DATA_RICH_FORMAT},
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen0_9;
}

static const DataModeLimits& _ModesBiasGen3()
{
    static const DataModeLimits __ModesBiasGen3 =
    {
        {GNA_INT8, GNA_INT16, GNA_INT32, GNA_DATA_RICH_FORMAT},
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen3;
}

static const DataModeLimits& _ModesBiasGen0_9Multibias()
{
    static const DataModeLimits __ModesBiasGen0_9Multibias =
    {
        {GNA_INT32 },
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen0_9Multibias;
}

static const DataModeLimits& _ModesBiasGen3Multibias()
{
    static const DataModeLimits __ModesBiasGen3Multibias =
    {
        {GNA_INT8, GNA_INT16, GNA_INT32},
        Gna2StatusXnnErrorBiasBytes
    };
    return __ModesBiasGen3Multibias;
}

const RangeLimits<>& limitsForWeightMultiplierElemMaxMultiplier()
{
    static const RangeLimits<> _limitsForWeightMultiplyElemMaxMultiply =
    {
        LayerCapabilities::InputElementsMultipllier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementsMultipllier,
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeightMultiplyElemMaxMultiply;
}

const RangeLimits<>& limitsForWeight()
{
    static const RangeLimits<> _limitsForWeight =
    {
        LayerCapabilities::limitsForInput(),
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeight;
}

const RangeLimits<>& limitsForBias()
{
    static const RangeLimits<> _limitsForBias =
    {
        LayerCapabilities::limitsForInput(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForBias;
}

static const RangeLimits<>& limitsForBiasGroupsMax()
{
    static const RangeLimits<> _limitsForBiasGroupsMax =
    {
        LayerCapabilities::limitsForInputGroupsMax(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForBiasGroupsMax;
}

static const RangeLimits<>& limitsForOutputRnn()
{
    static const RangeLimits<> _limitsForOutputRnn =
    {
        LayerCapabilities::RecurrentOutputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax,
        LayerCapabilities::RecurrentOutputElementCountMultiplier,
        Gna2StatusXnnErrorOutputVolume
    };
    return _limitsForOutputRnn;
}

static const RangeLimits<>& limitsForWeightRnnHeight()
{
    static const RangeLimits<> _limitsForWeightRnnHeight =
    {
        limitsForOutputRnn(),
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeightRnnHeight;
}

static const RangeLimits<>& limitsForWeightRnnWidth()
{
    static const RangeLimits<> _limitsForWeightRnnBasedOnInput =
    {
        LayerCapabilities::InputElementsMultipllier + LayerCapabilities::RecurrentOutputElementCountMultiplier,
        LayerCapabilities::InputElementCountMax + LayerCapabilities::InputElementCountMax,
        LayerCapabilities::InputElementsMultipllier,
        Gna2StatusXnnErrorWeightVolume
    };
    return _limitsForWeightRnnBasedOnInput;
}

static const RangeLimits<>& limitsForBiasRnn()
{
    static const RangeLimits<> _limitsForBiasRnn =
    {
        limitsForOutputRnn(),
        Gna2StatusXnnErrorBiasVolume
    };
    return _limitsForBiasRnn;
}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetInputComponentLimits(const gna_device_generation generation)
{
    static const OperationCapabilityMap operands =
    {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForInputShapeLegacy()},
            {GNA_DIM_W, limitsForInputGroupsMax()}},
            GetModes(InputOperandIndex, GNA_0_9)})},
        {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForInputShapeLegacy()},
            {GNA_DIM_W, limitsForInputGroupsMax()}},
            GetModes(InputOperandIndex, GNA_0_9)})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForInputShapeLegacy()},
            {GNA_DIM_W, limitsForInputGroupsMax()}},
            GetModes(InputOperandIndex, GNA_3_0)})},
    };
    return operands.at(generation);
}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetOutputComponentLimits(const gna_device_generation generation)
{
    static const OperationCapabilityMap operands =
    {
        {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForOutput()},
            {GNA_DIM_W, limitsForOutputGroupsMax()}},
            GetModes(OutputOperandIndex, GNA_0_9)})},
        {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForOutput()},
            {GNA_DIM_W, limitsForOutputGroupsMax()}},
            GetModes(OutputOperandIndex, GNA_0_9)})},
        {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_HW},
            {{GNA_DIM_H, limitsForOutput()},
            {GNA_DIM_W, limitsForOutputGroupsMax()}},
            GetModes(OutputOperandIndex, GNA_3_0)})},
    };
    return operands.at(generation);
}

const std::shared_ptr<ComponentLimits>& AffineLayerCapabilities::GetMBOutputComponentLimits(const gna_device_generation generation)
{
    auto multiBiasLimits = GetOutputComponentLimits(GNA_2_0);
    multiBiasLimits->Dimensions.at(GNA_DIM_H).Multipliers.at(Gna2DataTypeNone) = 8;
    static const OperationCapabilityMap operands =
    {
        {GNA_2_0, multiBiasLimits},
    };
    return operands.at(generation);
}

const FullCapabilitiesMap& AffineLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_AFFINE, {
                {GNA_0_9, GetInputComponentLimits(GNA_0_9)},
                {GNA_3_0, GetInputComponentLimits(GNA_3_0)},
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {GNA_0_9, GetInputComponentLimits(GNA_0_9)},
                {GNA_3_0, GetInputComponentLimits(GNA_3_0)},
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {GNA_2_0, GetInputComponentLimits(GNA_2_0)},
                {GNA_3_0, GetInputComponentLimits(GNA_3_0)},
            }},
            {INTEL_RECURRENT, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                     GetModes(InputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_3_0)})},
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_AFFINE, {
                {GNA_0_9, GetOutputComponentLimits(GNA_0_9)},
                {GNA_3_0, GetOutputComponentLimits(GNA_3_0)},
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {GNA_0_9, GetOutputComponentLimits(GNA_0_9)},
                {GNA_3_0, GetOutputComponentLimits(GNA_3_0)},
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {GNA_2_0, GetMBOutputComponentLimits(GNA_2_0)},
                {GNA_3_0, GetOutputComponentLimits(GNA_3_0)},
            }},
            {INTEL_RECURRENT, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputRnn()}}, // must be multiple 32 to keep 64B output buffer alignment
                    GetModes(OutputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputRnn()}}, // must be multiple 32 to keep 64B output buffer alignment
                    GetModes(OutputOperandIndex, GNA_3_0)})},
            }},
        }},
        {WeightOperandIndex,{
            {INTEL_AFFINE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},    // W - #inputs, H - #outputs
                    {{GNA_DIM_W, limitsForWeightMultiplierElemMaxMultiplier()},
                    {GNA_DIM_H, limitsForWeight()}},
                    _ModesWeightGen0_9()})}
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},    // W=H = #outputs
                    {{GNA_DIM_H, limitsForWeightMultiplierElemMaxMultiplier()}},
                    _ModesWeightGen0_9()})}
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},   // W - #inputs, H - #outputs
                    {{GNA_DIM_W, limitsForWeightMultiplierElemMaxMultiplier()},
                    {GNA_DIM_H, limitsForWeight()}},
                    _ModesWeightGen0_9()})}
            }},
            {INTEL_RECURRENT, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    { GNA_TENSOR_HW },
                    {{GNA_DIM_H, limitsForWeightRnnHeight()},
                    {GNA_DIM_W, limitsForWeightRnnWidth()}},
                    _ModesWeightGen0_9()})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_AFFINE, {
                {GNA_0_9,std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen3()})},
            }},
            {INTEL_AFFINE_DIAGONAL, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBias()}},
                    _ModesBiasGen3()})},
            }},
            {INTEL_AFFINE_MULTIBIAS, {
                {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForBias()},
                    {GNA_DIM_W, limitsForBiasGroupsMax()}},
                    _ModesBiasGen0_9Multibias()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForBias()},
                    {GNA_DIM_W, limitsForBiasGroupsMax()}},
                    _ModesBiasGen3Multibias()})}
            }},
            {INTEL_RECURRENT, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForOutputRnn()}},
                    _ModesBiasGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_H},
                    {{GNA_DIM_H, limitsForBiasRnn()}},
                    _ModesBiasGen3()})},
            }},
        }},
        { WeightScaleFactorOperandIndex,{
            {INTEL_AFFINE_MULTIBIAS, {
                {GNA_2_0, std::make_shared<TensorLimits>(TensorLimits{
            {GNA_TENSOR_H},
            {{GNA_DIM_H, {1, XNN_N_IN_ELEMS_MAX, 1, Gna2StatusXnnErrorBiasVolume}}},
            {{ GNA_DATA_RICH_FORMAT }, Gna2StatusXnnErrorBiasBytes }})}
        }},
    }}
    };

    return operands.at(operandIndex);
}
