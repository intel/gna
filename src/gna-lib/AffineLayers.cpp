/**
 @copyright (C) 2019-2021 Intel Corporation
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

#include "common.h"
#include "gna-api-types-xnn.h"


using namespace GNA;

void *AffineBaseLayer::GetGlobal2MBScratchpad()
{
    static void* ptr = nullptr;
    uint32_t sizeGranted;
    if (ptr == nullptr)
    {
        const auto status = Gna2MemoryAlloc(1 << 21, &sizeGranted, &ptr);
        if (status != Gna2StatusSuccess || ptr == nullptr)
        {
            Log->Error("Unsuccessful Scratchpad allocation\n");
        }
    }
    return ptr;
}

AffineBaseLayer::AffineBaseLayer(
    const nn_layer& layer, std::vector<TransformOperation> transforms,
    const BaseValidator& validatorIn) :
        Layer(layer, validatorIn, transforms, layer.pOutputsIntermediate)
{
}

AffineBaseLayer::AffineBaseLayer(
        const Gna2Operation& operation,
        const std::vector<TransformOperation> transforms,
        const BaseValidator& validatorIn) :
    Layer(operation, validatorIn, transforms, BaseAddress(GetGlobal2MBScratchpad()))
{
}

DataConfig AffineBaseLayer::GetDataMode() const
{
    auto affineTransform = static_cast<AffineFunction const *>(GetInputTransform());
    return AffineBaseLayer::getDataMode(affineTransform);
}

void AffineLayer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    AffineBaseLayer::UpdateKernelConfigs(layerConfiguration);
    auto activation = Transforms.Get<ActivationFunction>(ActivationTransform);
    if (activation)
    {
        auto const outputCount = layerConfiguration.ActList ?
            layerConfiguration.ActList->IndicesCount : Output.Dimensions.at('H');
        activation->UpdateActiveOutputCount(layerConfiguration.ConfigList,
                                            outputCount * Output.Dimensions.at('W'));
    }
}

AffineLayer::AffineLayer(const nn_layer& layer, const BaseValidator& validatorIn) :
    AffineBaseLayer(layer, { AffineTransform, ActivationTransform }, validatorIn)
{}

AffineLayer::AffineLayer(const Gna2Operation& operation, const BaseValidator& validatorIn) :
    AffineBaseLayer(operation, { AffineTransform, ActivationTransform }, validatorIn)
{
    if(operation.Type == Gna2OperationTypeElementWiseAffine)
    {
        ModelErrorHelper::ExpectEqual(Output.AsModelValue('H'), Input.AsModelValue('H'));
        ModelErrorHelper::ExpectEqual(Output.AsModelValue('W'), Input.AsModelValue('W'));
    }
}

Tensor const & AffineBaseLayer::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case ScratchpadOperandIndex:
        if (Transforms.Get(ActivationTransform))
        {
            return Output.ScratchPad;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    case WeightOperandIndex: //[[fallthrough]]
    case BiasOperandIndex: //[[fallthrough]]
    case WeightScaleFactorOperandIndex:
        return GetInputTransform()->GetOperand(operandIndex);
    case PwlOperandIndex:
        return getTransformOperand(ActivationTransform, 2);
    default:
        return Layer::GetOperand(operandIndex);
    }
}
