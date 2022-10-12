/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "LayerCapabilities.h"

#include "AffineLayerCapabilities.h"
#include "AuxiliaryCapabilities.h"
#include "ConvolutionalLayer2DCapabilities.h"
#include "GmmLayerCapabilities.h"

using namespace GNA;

const OperationCapabilityMap& LayerCapabilities::GetOperands(nn_operation operation, uint32_t operandIndex)
{
    switch (operation)
    {
    case INTEL_AFFINE:
    case INTEL_AFFINE_DIAGONAL:
    case INTEL_AFFINE_MULTIBIAS:
    case INTEL_RECURRENT:
        return AffineLayerCapabilities::GetOperands(operandIndex).at(operation);
    case INTEL_COPY:
    case INTEL_INTERLEAVE:
    case INTEL_DEINTERLEAVE:
        return AuxiliaryCapabilities::GetOperands(operandIndex).at(operation);
    case INTEL_CONVOLUTIONAL:
    case INTEL_CONVOLUTIONAL_2D:
    case INTEL_CONVOLUTIONAL_1D:
        return ConvolutionalLayer2DCapabilities::GetOperands(operandIndex).at(operation);
    case INTEL_GMM:
        return GmmLayerCapabilities::GetOperands(operandIndex).at(operation);
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}

const DataModeLimits& LayerCapabilities::GetCommonModes(uint32_t operandIndex, Gna2DeviceGeneration generation)
{
    static const std::map<uint32_t, std::map<Gna2DeviceGeneration, DataModeLimits>> modes =
    {
        {InputOperandIndex,
            {{Gna2DeviceGeneration0_9, {{Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration2_0, {{Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration3_0, {{Gna2DataTypeInt8, Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
            {Gna2DeviceGeneration3_1, {{Gna2DataTypeInt8, Gna2DataTypeInt16}, Gna2StatusXnnErrorInputBytes}},
        }},
        {OutputOperandIndex,
            {{Gna2DeviceGeneration0_9, {{Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration2_0, {{Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration3_0, {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
            {Gna2DeviceGeneration3_1, {{Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32}, Gna2StatusXnnErrorOutputBytes}},
        }},
        {BiasOperandIndex, {
            MakeModes<Gna2DeviceGeneration0_9, BiasOperandIndex>
                (Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
            MakeModes<Gna2DeviceGeneration2_0, BiasOperandIndex>
               (Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
            MakeModes<Gna2DeviceGeneration3_0, BiasOperandIndex>
               (MakeDataModesCartesian(
                    {Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias},
                    { Gna2TensorModeDefault, Gna2TensorModeDisabled })),
            MakeModes<Gna2DeviceGeneration3_1, BiasOperandIndex>
               (Gna2DataTypeInt8, Gna2DataTypeInt16, Gna2DataTypeInt32, Gna2DataTypeCompoundBias),
        }},
    };
    return modes.at(operandIndex).at(generation);
}

constexpr StaticCaps LayerCapabilities::Input;
constexpr StaticCaps LayerCapabilities::InputGroupMax;
constexpr RangeLimits<uint32_t> LayerCapabilities::LegacyInputs;
constexpr StaticCaps LayerCapabilities::InputEqual1;
constexpr StaticCaps LayerCapabilities::Input1D;
constexpr StaticCaps LayerCapabilities::WeightMultiplier;
constexpr StaticCaps LayerCapabilities::OutputRnn;
