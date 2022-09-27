/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "Layer.h"

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
    static void RelaseGlobal2MBScrachpad();

protected:
    AffineBaseLayer(
            const Gna2Operation& operation, std::vector<TransformOperation> transforms,
            const LayerValidator& validatorIn);

    template<typename TransformFunction>
    void setDataMode(TransformFunction const & transform, bool isActivationDisabled)
    {
        auto weightMode = transform.Weights->Mode;
        auto biasMode = transform.Biases->Mode;
        dataConfig = DataConfig{ Input.Mode, weightMode, biasMode, Output.Mode, isActivationDisabled };
    }

private:
    static void* scratchPad;
};

class AffineLayer : public AffineBaseLayer
{
public:
    AffineLayer(const Gna2Operation& operation, const LayerValidator& validatorIn);
    virtual ~AffineLayer() = default;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;
};

}
