/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "AffineLayers.h"

#include "ActivationHelper.h"
#include "ActiveList.h"
#include "Address.h"
#include "KernelArguments.h"
#include "LayerConfiguration.h"
#include "LayerOutput.h"
#include "Logger.h"
#include "Tensor.h"

#include "gna2-common-api.h"
#include "gna2-memory-api.h"
#include "gna2-model-export-api.h"

using namespace GNA;

void *AffineBaseLayer::GetGlobal2MBScratchpad()
{
    static void* ptr = nullptr;
    uint32_t sizeGranted;
    if (ptr == nullptr)
    {
        auto status = Gna2MemoryAlloc(1 << 21, &sizeGranted, &ptr);
        if (status != Gna2StatusSuccess || ptr == nullptr)
        {
            Log->Error("Unsuccessful Scratchpad allocation\n");
        }
        status = Gna2MemorySetTag(ptr, Gna2MemoryTagScratch);
        if (status != Gna2StatusSuccess)
        {
            Log->Error("Unsuccessful Scratchpad tagging\n");
        }
    }
    return ptr;
}

AffineBaseLayer::AffineBaseLayer(
        const Gna2Operation& operation,
        const std::vector<TransformOperation> transforms,
        const LayerValidator& validatorIn) :
    Layer(operation, validatorIn, transforms, BaseAddress(LayerOutput::getScratchpadForOperation(validatorIn.Operation)))
{
}

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);
    auto const activation = Transforms.GetOptional<ActivationFunction>(ActivationTransform);
    if (activation)
    {
        auto const outputCount = layerConfiguration.ActList ?
            layerConfiguration.ActList->IndicesCount : Output.Dimensions.at('H');
        activation->UpdateActiveOutputCount(layerConfiguration.ConfigList,
                                            outputCount * Output.Dimensions.at('W'));
    }
}

AffineLayer::AffineLayer(const Gna2Operation& operation, const LayerValidator& validatorIn) :
    AffineBaseLayer(operation, { AffineTransform, ActivationTransform }, validatorIn)
{
    if(operation.Type == Gna2OperationTypeElementWiseAffine)
    {
        ModelErrorHelper::ExpectEqual(Output.at('H'), Input.at('H'));
        ModelErrorHelper::ExpectEqual(Output.at('W'), Input.at('W'));
    }
    auto const & affineTransform = GetInputTransform<AffineFunction>();
    auto const activation = Transforms.GetOptional<ActivationFunction>(ActivationTransform);
    setDataMode(affineTransform, activation == nullptr);
}

Tensor const & AffineBaseLayer::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        if (!dataConfig.IsActivationDisabled)
        {
            return Output.ScratchPad;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case WeightOperandIndex:
    case BiasOperandIndex:
    case WeightScaleFactorOperandIndex:
        return GetInputTransform().GetOperand(operandIndex);
    case PwlOperandIndex:
        return getTransformOperand(ActivationTransform, 2);
    default:
        return Layer::GetOperand(operandIndex);
    }
}
