/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "AccelerationDetector.h"
#include "DataMode.h"
#include "Shape.h"
#include "ModelError.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"

namespace GNA
{
class OperationConfig
{
public:
    OperationConfig(const Gna2Operation& apiOperation);

    static Gna2PoolingMode GetPoolingMode(const Gna2Operation& operation);
    static bool IsCNN1D(const Gna2Operation & operation);

    static bool IsMultibias(const Gna2Operation& operation);

    static Gna2OperationType GetOperationType(const Gna2Operation& operation);

    kernel_op GetKernelOperation() const;
    TransformOperation GetTransformOperation() const;

    Gna2Tensor GetEnabledOperand(uint32_t index) const;

    template<typename T>
    T GetParameterAs(uint32_t index) const
    {
        Expect::NotNull(Operation);
        return ModelWrapper::GetParameter<T>(*Operation, index);
    }

    static std::unique_ptr<const Component> CreateCnnComponent(const Shape& shape,
        const LayerValidator& validator, const FullCapabilitiesMap & caps, const uint32_t parameterIndex);

    Gna2OperationType OperationType;
    Gna2Tensor WeightsTensor;
    Gna2Tensor FiltersTensor;
    Shape ConvolutionStride;
    Shape ZeroPadding;
    Shape PoolingWindow;
    Shape PoolingStride;
    Gna2PoolingMode Mode;
    Gna2Tensor BiasesTensor;
    Gna2BiasMode BiasMode = Gna2BiasModeDefault;
    uint32_t FeedbackDelay;
    Gna2Tensor WeightScalesTensor = ModelWrapper::GetDisabledOperand();
    uint32_t BiasVectorIndex;

    Gna2Operation const * const Operation;

protected:
    void InitOperationConfig(const Gna2Operation& operation)
    {
        OperationType = GetOperationType(operation);
        BiasesTensor = GetBiases(operation);

        if (isAffine(operation))
        {
            WeightsTensor = GetWeights(operation);
            if (IsMultibias(operation))
            {
                InitMultibias(operation);
            }
        }
        if (isRecurrent(operation))
        {
            FeedbackDelay = GetFeedbackDelay(operation);
        }
        if (isCNN2D(operation))
        {
            FiltersTensor = GetFilters(operation);
            ConvolutionStride = GetStride(operation);
            ZeroPadding = GetZeroPadding(operation);
            BiasMode = GetBiasMode(operation);
            if (hasPooling(operation))
            {
                InitPooling(operation);
            }
            else
            {
                Mode = Gna2PoolingModeDisabled;
            }
        }
    }
    void InitPooling(const Gna2Operation& operation);

    void InitMultibias(const Gna2Operation& operation);

private:
    static Shape TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex);

    static Gna2Tensor GetWeights(const Gna2Operation& operation);
    static Gna2Tensor GetFilters(const Gna2Operation& operation);

    static Gna2BiasMode GetBiasMode(const Gna2Operation& operation);
    static Gna2Tensor GetBiases(const Gna2Operation& operation);

    static Shape GetStride(const Gna2Operation& operation);

    static uint32_t GetFeedbackDelay(const Gna2Operation& operation);

    static Shape GetZeroPadding(const Gna2Operation& operation);

    static Shape GetShapeParameterOfMaximal2Dimensions(const Gna2Operation& operation, uint32_t parameterIndex);

    static bool hasPooling(const Gna2Operation& operation);

    static bool isCNN2D(const Gna2Operation& operation);

    static bool isAffine(const Gna2Operation& operation);

    static bool isRecurrent(const Gna2Operation& operation);
};

}
