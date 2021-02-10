/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "gna2-inference-impl.h"

#include <memory>

#include "DeviceLayerSupport.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Transform.h"
#include "TransformMap.h"

#include "Address.h"
#include "KernelArguments.h"
#include "Validator.h"

#include "common.h"
#include "gna-api.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GNA
{

class BufferMap;
struct LayerConfiguration;

class AbstractOperation
{
public:
    const nn_operation Operation;
    const Gna2OperationType OperationNew;
protected:
    AbstractOperation(const Gna2Operation& operation, const BaseValidator& validator) :
        Operation{ toLegacy(operation, validator) },
        OperationNew{ operation.Type }
    {
    }

    AbstractOperation(const nn_layer& layer, const BaseValidator& validator) :
        Operation{ layer.operation },
        OperationNew{ fromLegacy(layer.operation) }
    {
        UNREFERENCED_PARAMETER(validator);
    }
private:
    static nn_operation toLegacy(const Gna2Operation& operation, const BaseValidator& validator);
    static Gna2OperationType fromLegacy(const nn_operation& layerType);
};

class Layer : public AbstractOperation
{
public:
    static std::unique_ptr<Layer> Create(const nn_layer& layer, const BaseValidator& validator);
    static std::unique_ptr<Layer> Create(const Gna2Operation& operation, const BaseValidator& validator);

    template<typename X = const Layer> X* Get() const
    {
        return static_cast<const X*>(this);
    }

    virtual ~Layer() = default;
    std::function<void(AccelerationMode accel, ExecutionConfig const & executionConfig)> ComputeHidden;
    std::function<void(LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)> Compute;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const;
    virtual DataConfig GetDataMode() const;

    TransformList Transforms;

    BaseTransform const * GetInputTransform() const
    {
        return inputTransform;
    };
    BaseTransform const * GetOutputTransform() const
    {
        return outputTransform;
    };

    virtual Tensor const & GetOperand(uint32_t operandIndex) const;
    Tensor const * TryGetOperand(uint32_t operandIndex) const;

    uint32_t TryGetOperandSize(uint32_t operandIndex) const;

    bool Is1BInputAnd2BWeight() const
    {
        return has1BInputAnd2BWeight;
    }

    virtual void VerifyHas1BInputAnd2BWeight();

protected:
    std::unique_ptr<const LayerValidator> validator;

public:
    const LayerInput Input;
    const LayerOutput Output;

protected:
    template <class T>
    Layer(const T& layer, const BaseValidator& validatorIn,
        const std::vector<TransformOperation>& transforms,
        const BaseAddress& intermediateBuffer) :
        AbstractOperation{ layer, validatorIn },
        validator{ std::make_unique<const LayerValidator>(validatorIn, Operation) },
        Input{ layer, *validator },
        Output{ layer, *validator }
    {
        Expect::InRange<uint32_t>(Operation, 0, LAYER_OPERATION_TYPE_COUT - 1, Gna2StatusXnnErrorLyrOperation);
        if (false == transforms.empty())
        {
            auto&& commonConfig = TransformFactoryConfig(&Input, &Output, Output.Mode, intermediateBuffer,
                layer, *validator);
            const OperationConfig operationConfig{ layer };
            initTransforms(transforms, commonConfig, operationConfig);
        }

        initComputeFunctions();
    }

    void initTransforms(const std::vector<TransformOperation>& transforms, TransformFactoryConfig& commonConfig,
        const OperationConfig& operationConfig);

    void initComputeFunctions();

    void compute(const LayerConfiguration* layerConfiguration,
        AccelerationMode accel, ExecutionConfig const & execution) const;

    Tensor const & getTransformOperand(TransformOperation operation, uint32_t operandIndex) const;

    BaseTransform const * inputTransform = nullptr;
    BaseTransform * outputTransform = nullptr;

private:
    void addBufferAs(const BufferMap& source, uint32_t sourceType,
        BufferMap& destination, uint32_t destinationType) const;

    bool has1BInputAnd2BWeight = false;
    bool is1BInputAnd2BWeightVerified = false;
};

}
