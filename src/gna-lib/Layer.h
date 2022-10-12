/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-inference-impl.h"

#include "DeviceLayerSupport.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Transform.h"
#include "TransformMap.h"
#include "Address.h"
#include "KernelArguments.h"
#include "Validator.h"

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

    static nn_operation toLegacy(const Gna2Operation& operation, const BaseValidator& validator);

protected:
    AbstractOperation(const Gna2Operation& operation, const LayerValidator& validator) :
        Operation{ validator.Operation },
        OperationNew{ operation.Type }
    {
    }

    static Gna2OperationType fromLegacy(const nn_operation& layerType);
};

class Layer : public AbstractOperation
{
public:
    static std::unique_ptr<Layer> Create(const Gna2Operation& operation, const BaseValidator& validator);

    template<typename X = const Layer> X* Get() const
    {
        return static_cast<const X*>(this);
    }

    virtual ~Layer() = default;

    std::function<void(AccelerationMode accel, ExecutionConfig const & executionConfig)> ComputeHidden;
    std::function<void(LayerConfiguration &layerConfiguration, AccelerationMode accel, ExecutionConfig const & executionConfig)> Compute;

    virtual void UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const;
    const DataConfig GetDataMode() const;

    TransformList Transforms;

    template<typename TransformType = BaseTransform>
    TransformType const & GetInputTransform() const
    {
        Expect::NotNull(inputTransform);
        return *(reinterpret_cast<TransformType const *>(inputTransform));
    }

    template<typename TransformType = BaseTransform>
    TransformType const & GetOutputTransform() const
    {
        Expect::NotNull(outputTransform);
        return *(reinterpret_cast<TransformType *>(outputTransform));
    }

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
    Layer(const Gna2Operation& operation, const LayerValidator& validatorIn,
        const std::vector<TransformOperation>& transforms,
        const BaseAddress& intermediateBuffer) :
        AbstractOperation{ operation, validatorIn },
        validator{ std::make_unique<const LayerValidator>(validatorIn) },
        Input{ operation, *validator },
        Output{ operation, *validator }
    {
        Expect::InRange<uint32_t>(Operation, 0, LAYER_OPERATION_TYPE_COUT - 1, Gna2StatusXnnErrorLyrOperation);

        if (!transforms.empty())
        {
            auto&& commonConfig = TransformFactoryConfig(&Input, &Output, Output.Mode, intermediateBuffer,
                operation, *validator);
            const OperationConfig operationConfig{ operation };
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
    DataConfig dataConfig;

private:
    void addBufferAs(const BufferMap& source, uint32_t sourceType,
        BufferMap& destination, uint32_t destinationType) const;

    bool has1BInputAnd2BWeight = false;
    bool is1BInputAnd2BWeightVerified = false;
};

}
