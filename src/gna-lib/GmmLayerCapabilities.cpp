/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "GmmLayerCapabilities.h"

#include "DataMode.h"
#include "Capabilities.h"
#include "Tensor.h"

#include "LayerCapabilities.h"

using namespace GNA;


const FullCapabilitiesMap& GmmLayerCapabilities::GetOperands(uint32_t operandIndex)
{
    static const ComponentFullCapabilityMap operands =
    {
        {InputOperandIndex,{
            {INTEL_GMM, {
                {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},
                    {{GNA_DIM_H, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorInputVolume}},
                    {GNA_DIM_W, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
                    {{ GNA_UINT8}, Gna2StatusXnnErrorInputBytes }})}
                }},
        }},
        {OutputOperandIndex, {
            {INTEL_GMM, {
                {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW}, // H - GMM States, W - grouping
                    {{GNA_DIM_W, {1, XNN_N_GROUP_MAX, 1, Gna2StatusXnnErrorOutputVolume}},
                    {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusXnnErrorOutputVolume}}},
                    {{GNA_UINT32, GNA_DATA_ACTIVATION_DISABLED }, Gna2StatusXnnErrorOutputBytes}})}
            }},
        }},
        {WeightOperandIndex, {
            {INTEL_GMM, {
                {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HWD },                  // H - GMM states, W - #mixtures, D - #inputs
                    {{GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}},
                    {GNA_DIM_W, {1, GMM_MIXTURE_COMP_COUNT_MAX, 1, Gna2StatusGmmBadMixCnum}},
                    {GNA_DIM_D, {GMM_FV_ELEMENT_COUNT_MIN, GMM_FV_ELEMENT_COUNT_MAX, GMM_FV_ELEMENT_COUNT_MULTIPLE_OF, Gna2StatusBadFeatLength}}},
                    {{GNA_UINT8, GNA_UINT16}, Gna2StatusGmmBadMode},
                    {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadVarsAlign}})}
            }},
        }},
        {BiasOperandIndex,{
            {INTEL_GMM, {
                {GMM_DEVICE, std::make_shared<TensorLimits>(TensorLimits{
                    {GNA_TENSOR_HW},                   // H - GMM states, W - #mixtures
                    {{GNA_DIM_W, {1, GMM_MIXTURE_COMP_COUNT_MAX, 2, Gna2StatusGmmBadMixCnum}},
                    {GNA_DIM_H, {1, GMM_STATES_COUNT_MAX, 1, Gna2StatusGmmBadNumGmm}}},
                    {{GNA_UINT32}, Gna2StatusGmmBadMode},
                    {GMM_MEM_ALIGNMENT, Gna2StatusGmmBadGconstAlign}})}
            }}
        }}
    };
    return operands.at(operandIndex);

}

