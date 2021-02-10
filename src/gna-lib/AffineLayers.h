/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "Layer.h"

#include "common.h"

#include <memory>

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class AffineBaseLayer : public Layer
{
public:
    virtual ~AffineBaseLayer() = default;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

    static void *GetGlobal2MBScratchpad();

protected:
    AffineBaseLayer(
            const nn_layer& layer, std::vector<TransformOperation> transforms,
            const BaseValidator& validatorIn);

    AffineBaseLayer(
            const Gna2Operation& operation, std::vector<TransformOperation> transforms,
            const BaseValidator& validatorIn);

    virtual DataConfig GetDataMode() const override;

    template<typename TransformFunction>
    DataConfig getDataMode(TransformFunction transform) const
    {
        auto weightMode = transform->Weights->Mode.Value;
        auto biasMode = transform->Biases->Mode.Value;
        return DataConfig(Input.Mode, weightMode, biasMode, Output.Mode);
    }
};

class AffineLayer : public AffineBaseLayer
{
public:
    AffineLayer(const nn_layer& layer, const BaseValidator& validatorIn);
    AffineLayer(const Gna2Operation& operation, const BaseValidator& validatorIn);
    virtual ~AffineLayer() = default;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;
};

}
