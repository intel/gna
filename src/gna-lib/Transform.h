/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "BufferMap.h"
#include "DataMode.h"
#include "GnaException.h"
#include "KernelArguments.h"
#include "LayerConfiguration.h"
#include "ModelWrapper.h"
#include "Tensor.h"
#include "XnnKernel.h"
#include "common.h"

#include "gna-api.h"

#include <memory>
#include <stdexcept>

namespace GNA
{
class LayerValidator;

struct TransformFactoryConfig
{
    const Tensor * input;
    const Tensor * output;
    const DataMode outputMode;
    const BaseAddress outputBuffer;
    const LayerValidator& validator;

    template<class T>
    TransformFactoryConfig(const Tensor *inputIn, const Tensor *outputIn, DataMode outputModeIn,
        BaseAddress outputBufferIn, const T& operation, const LayerValidator& validatorIn) :
        input{ inputIn }, output{ outputIn }, outputMode{ outputModeIn }, outputBuffer{ outputBufferIn },
        validator{ validatorIn }
    {
        InitActivation(operation);
    }

    TransformFactoryConfig() = delete;
    bool HasMandatoryActivation() const;
    bool IsActivationNotSupported() const;
    Gna2Tensor GetActivation() const;
    static Gna2Tensor GetActivation(const void * layerDetails, nn_operation operationType);

protected:
    void InitActivation(const nn_layer& layer);
    void InitActivation(const Gna2Operation& operation);
private:

    bool HasMandatoryActivation(const void * layerDetails) const;

    static bool HasMandatoryActivation(const Gna2Operation& operation);

    static Gna2Tensor GetActivation(const Gna2Operation& operation);

    bool mandatoryActivation;
    Gna2Tensor activation;
};

template<typename KernelType>
struct BaseTransformConfig
{
    const Tensor * input;
    const Tensor * output;
    const DataMode outputMode;
    const BaseAddress& outputBuffer; // transform output buffer, usually layer intermediate buffer
    const LayerValidator& validator;
    const KernelMap<KernelType>& kernels;
    const KernelMap<KernelType>* kernelsAl;

    BaseTransformConfig(const TransformFactoryConfig& config, const KernelMap<KernelType>& kernelsIn,
        const KernelMap<KernelType>* kernelsAlIn = nullptr) :
        input{ config.input },
        output{ config.output },
        outputMode{ config.output->Mode },
        outputBuffer{ config.outputBuffer },
        validator{ config.validator },
        kernels{ kernelsIn },
        kernelsAl{ kernelsAlIn }
    {}

    BaseTransformConfig() = delete;
};

class BaseTransform
{
public:
    virtual ~BaseTransform() = default;

    virtual void Compute(AccelerationMode accel, LayerConfiguration const * layerConfiguration,
        ExecutionConfig const & execution) const = 0;

    virtual void UpdateConfigBuffers(std::unique_ptr<BaseConfig> configs[TransformOperationCount],
        const BufferMap& buffers) const = 0;

    virtual void ValidateActiveList(ActiveList const & activeList) const
    {
        UNREFERENCED_PARAMETER(activeList);
        throw GnaException(Gna2StatusActiveListIndicesInvalid);
    }

    virtual void SetOutput(const BaseAddress& outputBuffer) = 0;
    virtual Tensor const & GetOperand(uint32_t operandIndex) const;
    template<class T>
    static T const & GetOperandIfExistOrThrow(std::unique_ptr<T> const & operand)
    {
        if (operand)
        {
            return *operand;
        }
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }

    virtual bool Is1D() const
    {
        return false;
    }

    const Tensor * const Input;
    std::unique_ptr<Tensor> Output;
    TransformOperation const Operation;

protected:
    BaseTransform(TransformOperation operation, Tensor const * input) :
        Input{ input },
        Operation{ operation }
    {};
    BaseTransform(const BaseTransform&) = delete;
    BaseTransform(const BaseTransform&&) = delete;
};

template<typename TransformType, typename KernelType>
class Transform : public BaseTransform
{
public:
    virtual void Compute(AccelerationMode accel,
        LayerConfiguration const * layerConfiguration,
        ExecutionConfig const & execution) const override
    {
        auto executionConfig = createExecutionConfig(layerConfiguration, execution);
        updateExecutionKernelConfig(*executionConfig);
        try
        {
            kernels->at(accel)(executionConfig.get());
        }
        catch (const std::out_of_range&)
        {
            throw GnaException(Gna2StatusNotImplemented);
        }
    }

    virtual void UpdateConfigBuffers(std::unique_ptr<BaseConfig> configs[TransformOperationCount],
        const BufferMap& buffers) const override
    {
        auto* config = GetConfig(configs);
        for (auto const & buff : buffers)
        {
            auto const result = config->SetBuffer(buff.first, buff.second);
            if (!result)
            {
                throw GnaException(Gna2StatusIdentifierInvalid);
            }
        }
    }

    // set output when transform is final layer transform and uses user provided layer output buffer
    virtual void SetOutput(const BaseAddress& outputBuffer) override
    {
        Output->UpdateBuffer(outputBuffer);
        hiddenConfig->SetBuffer(OutputOperandIndex, outputBuffer);
    }

protected:
    Transform(TransformOperation operation, const KernelMap<KernelType>* kernelsIn, Tensor const * input) :
        BaseTransform{ operation, input },
        kernels{ kernelsIn }
    {};
    Transform(const Transform&) = delete;
    Transform(const Transform&&) = delete;
    virtual ~Transform() = default;

    inline KernelConfig<TransformType>* GetConfig(
        std::unique_ptr<BaseConfig> configs[TransformOperationCount]) const
    {
        auto& config = configs[Operation];
        if (!config)
        {
            config = std::make_unique<KernelConfig<TransformType>>(*hiddenConfig);
        }
        return static_cast<KernelConfig<TransformType>*>(config.get());
    }

    const KernelMap<KernelType>* kernels;

    std::unique_ptr<KernelConfig<TransformType>> hiddenConfig;

    inline std::unique_ptr<ExecutionKernelConfig<TransformType>> createExecutionConfig(
        const LayerConfiguration* layerConfiguration, ExecutionConfig const & execution) const
    {
        if (nullptr == layerConfiguration)
        {
            return std::make_unique<ExecutionKernelConfig<TransformType>>(
                hiddenConfig.get(), execution);
        }
        else
        {
            return std::make_unique<ExecutionKernelConfig<TransformType>>(
                static_cast<KernelConfig<TransformType>*>(layerConfiguration->ConfigList[Operation].get()),
                execution);
        }
    }

    virtual void updateExecutionKernelConfig(ExecutionKernelConfig<TransformType> & config) const
    {
        UNREFERENCED_PARAMETER(config);
    }

    static void setSoftwareScratchPad(ExecutionKernelConfig<TransformType> & config)
    {
        if (nullptr != config.Intermediate && nullptr != config.Intermediate->cnnFusedBuffer)
        {
            if (nullptr == config.RequestConfig->Inputs)
            {
                config.RequestConfig->SetBuffer(GNA::InputOperandIndex, config.Intermediate->cnnFusedBuffer);
            }
            if (nullptr == config.RequestConfig->Outputs)
            {
                config.RequestConfig->SetBuffer(GNA::OutputOperandIndex, config.Intermediate->cnnFusedBuffer);
            }
        }
    }
};

template<typename TransformType, typename KernelType, typename KernelTypeAl>
class TransformAl : public Transform<TransformType, KernelType>
{
public:
    TransformAl(const TransformAl&) = delete;
    TransformAl(const TransformAl&&) = delete;
    virtual ~TransformAl() = default;

    virtual void Compute(AccelerationMode accel,
        LayerConfiguration const * layerConfiguration,
        ExecutionConfig const & execution) const override
    {
        auto executionConfig = Transform<TransformType, KernelType>::createExecutionConfig(
            layerConfiguration, execution);
        try
        {
            if (layerConfiguration != nullptr && layerConfiguration->ActList)
            {
                kernelsAl->at(accel)(executionConfig.get(), AffineConfigAl{
                                    layerConfiguration->ActList->Indices,
                                    layerConfiguration->ActList->IndicesCount });
            }
            else
            {
                Transform<TransformType, KernelType>::kernels->at(accel)(executionConfig.get());
            }
        }
        catch (const std::out_of_range&)
        {
            throw GnaException(Gna2StatusNotImplemented);
        }
    }

protected:
    TransformAl(TransformOperation operation,
        const KernelMap<KernelType>* kernelsIn,
        const KernelMap<KernelTypeAl>* kernelsAlIn,
        Tensor const * input) :
        Transform<TransformType, KernelType>{ operation, kernelsIn, input },
        kernelsAl{ kernelsAlIn }
    {};

    const KernelMap<KernelTypeAl>* kernelsAl;
};


}

