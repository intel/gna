/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "AuxiliaryCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "Tensor.h"


using namespace GNA;

static const DataModeLimits& _ModesOutputCopyGen0_9()
{
    static const DataModeLimits __ModesOutputCopyGen0_9 =
    {
        {GNA_INT16},
        Gna2StatusXnnErrorOutputBytes
    };
    return __ModesOutputCopyGen0_9;
}

static const DataModeLimits& _ModesOutputCopyGen3()
{
    static const DataModeLimits __ModesOutputCopyGen3 =
    {
        {GNA_INT8, GNA_INT16},
        Gna2StatusXnnErrorOutputBytes
    };
    return __ModesOutputCopyGen3;
}

const FullCapabilitiesMap& AuxiliaryCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_COPY, {
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
            {INTEL_INTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputGroupsMax()},
                    {GNA_DIM_W, limitsForInputShapeLegacy()}},
                    GetModes(InputOperandIndex, GNA_3_0)})}
            }},
            {INTEL_DEINTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputShapeLegacy()},
                    {GNA_DIM_W, limitsForInputGroupsMax()}},
                    GetModes(InputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForInputShapeLegacy()},
                    {GNA_DIM_W, limitsForInputGroupsMax()}},
                    _ModesOutputCopyGen3()})}
            }},
        }},
        {OutputOperandIndex,{
            {INTEL_COPY, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputShapeLegacy()}},
                    _ModesOutputCopyGen0_9()})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputShapeLegacy()}},
                    _ModesOutputCopyGen3()})}
            }},
            {INTEL_DEINTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputShapeLegacy()}},
                    GetModes(OutputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H,limitsForOutputGroupsMax()},
                    {GNA_DIM_W, limitsForOutputShapeLegacy()}},
                    GetModes(OutputOperandIndex, GNA_3_0)})},
            }},
            {INTEL_INTERLEAVE, {
                {GNA_0_9, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputShapeLegacy()},
                    {GNA_DIM_W, limitsForOutputGroupsMax()}},
                    GetModes(OutputOperandIndex, GNA_0_9)})},
                {GNA_3_0, std::make_shared<TensorLimits>(
                    TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, limitsForOutputShapeLegacy()},
                    {GNA_DIM_W, limitsForOutputGroupsMax()}},
                    GetModes(OutputOperandIndex, GNA_3_0)})},
            }},
        }},
    };

    return operands.at(operandIndex);
}
