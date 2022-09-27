/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "ActivationFunction.h"
#include "ActivationHelper.h"
#include "ConvolutionalFunctions.h"
#include "Layer.h"
#include "PoolingFunctions.h"

#include <memory>

struct ExecutionConfig;

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class CnnLayer : public Layer
{
public:
    CnnLayer(const Gna2Operation& apiLayer, const LayerValidator& validatorIn);
    virtual ~CnnLayer() = default;
    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    static bool IsForced(const Gna2Operation& operation);

    std::unique_ptr<const ConvolutionFunction> Convolution;
    std::unique_ptr<const PoolingFunction> Pooling;
    std::unique_ptr<const ActivationFunction> Activation;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

protected:
    void Init();

    std::unique_ptr<const ConvolutionFunction> GetConvolution(const Gna2Operation& apiOperation) const
    {
        const Tensor* convolutionOutput = &Output;

        if (ActivationHelper::IsEnabled(apiOperation))
        {
            convolutionOutput = &Output.ScratchPad;
        }
        return ConvolutionFunction::Create(&Input, convolutionOutput,
            apiOperation, *validator);
    }

    void ExpectValid() const;

    std::unique_ptr<const PoolingFunction> GetPooling(const Gna2Operation & apiOperation) const;

private:
    void computePool(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHiddenPool(AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHidden(AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHiddenPwl(AccelerationMode accel, ExecutionConfig const & execution) const;
    void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;
    void computePwl(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;

    static const Gna2Operation& getDetails(const Gna2Operation& operation);
};

}
