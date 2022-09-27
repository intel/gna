/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "GmmLayerCapabilities.h"

#include "Capabilities.h"
#include "DataMode.h"
#include "LayerCapabilities.h"
#include "Tensor.h"

using namespace GNA;

const FullCapabilitiesMap& GmmLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, BatchSizeMax, 1, Gna2StatusXnnErrorInputVolume}},
                    {GNA_DIM_W, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
                    {{ Gna2DataTypeUint8}, Gna2StatusXnnErrorInputBytes }})}
                }},
        }},
        {OutputOperandIndex, {
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW}, // H - GMM States, W - grouping
                    {{GNA_DIM_W, {1, BatchSizeMax, 1, Gna2StatusXnnErrorOutputVolume}},
                    {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
                    {{Gna2DataTypeUint32 }, Gna2StatusXnnErrorOutputBytes}})}
            }},
        }},
        {WeightOperandIndex, {
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HWD },                  // H - GMM states, W - #mixtures, D - #inputs
                    {{GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}},
                    {GNA_DIM_W, {1, GMM_MIXTURE_COMP_COUNT_MAX, 1, Gna2StatusGmmBadMixCnum}},
                    {GNA_DIM_D, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
                    {{Gna2DataTypeUint8, Gna2DataTypeUint16}, Gna2StatusGmmBadMode},
                    {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadVarsAlign}})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_GMM, {
                {Gna2DeviceGenerationGmm, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},                   // H - GMM states, W - #mixtures
                    {{GNA_DIM_W, {1, GMM_MIXTURE_COMP_COUNT_MAX, 2, Gna2StatusGmmBadMixCnum}},
                    {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}}},
                    {{Gna2DataTypeUint32}, Gna2StatusGmmBadMode},
                    {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadGconstAlign}})}
            }}
        }}
    };
    return operands.at(operandIndex);

}

