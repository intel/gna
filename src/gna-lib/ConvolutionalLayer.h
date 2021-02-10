/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "ActivationFunction.h"
#include "ActivationHelper.h"
#include "ConvolutionalFunctions.h"
#include "Layer.h"
#include "PoolingFunctions.h"

#include "common.h"

#include <memory>

struct ExecutionConfig;

namespace GNA
{
class BaseValidator;
struct LayerConfiguration;

class CnnLayer : public Layer
{
public:
    template<class T>
    CnnLayer(const T& apiLayer, const BaseValidator& validatorIn) :
        Layer(apiLayer, validatorIn, {}, BaseAddress())
    {
        ExpectValid();
        Convolution = GetConvolution(getDetails(apiLayer));
        Activation = ActivationFunction::Create({ &Output.ScratchPad, &Output, Output.Mode, Output.Buffer,
            apiLayer, *validator });
        Pooling = GetPooling(apiLayer);
        Init();
    }

    virtual ~CnnLayer() = default;
    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const override;

    static std::unique_ptr<Layer> CreateEnforced(const Gna2Operation& operation, const BaseValidator& base_validator);

    static bool IsForced(const Gna2Operation& operation);

    std::unique_ptr<const ConvolutionFunction> Convolution;
    std::unique_ptr<const PoolingFunction> Pooling;
    std::unique_ptr<const ActivationFunction> Activation;

    virtual Tensor const & GetOperand(uint32_t operandIndex) const override;

protected:
    void Init();

    template<class T>
    std::unique_ptr<const ConvolutionFunction> GetConvolution(const T& apiOperation) const
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
    std::unique_ptr<const PoolingFunction> GetPooling(const nn_layer & layer) const;

    virtual DataConfig GetDataMode() const override;

private:
    void computePool(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHiddenPool(AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHidden(AccelerationMode accel, ExecutionConfig const & execution) const;
    void computeHiddenPwl(AccelerationMode accel, ExecutionConfig const & execution) const;
    void compute(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;
    void computePwl(const LayerConfiguration& layerConfiguration, AccelerationMode accel, ExecutionConfig const & execution) const;

    static const nn_layer_conv& getDetails(const nn_layer& cnn1DLayer);
    static const Gna2Operation& getDetails(const Gna2Operation& operation);
};

}
