/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "AccelerationDetector.h"
#include "DataMode.h"
#include "Shape.h"
#include "ModelError.h"
#include "ModelWrapper.h"

#include "gna2-model-api.h"
#include "common.h"

namespace GNA
{
class OperationConfig
{
public:
    OperationConfig(const nn_layer& layer);

    OperationConfig(const Gna2Operation& apiOperation);

    static Gna2PoolingMode GetPoolingMode(const Gna2Operation& operation);
    static bool IsCNN1D(const Gna2Operation & operation);

    static bool IsMultibias(const nn_layer& layer);
    static bool IsMultibias(const Gna2Operation& operation);

    static Gna2OperationType GetOperationType(const Gna2Operation& operation);
    static Gna2OperationType GetOperationType(const nn_layer& layer);

    kernel_op GetKernelOperation() const;
    TransformOperation GetTransformOperation() const;

    Gna2Tensor GetEnabledOperand(uint32_t index) const;

    template<typename T>
    T GetParameterAs(uint32_t index) const
    {
        Expect::NotNull(Operation);
        return ModelWrapper::GetParameter<T>(*Operation, index);
    }

    template<typename Target, typename Source>
    static std::unique_ptr<const Target> CreateCnnTarget(
        const Source& source, const LayerValidator& validator, const FullCapabilitiesMap& caps)
    {
        try
        {
            // 1D CNN in new arch
            auto const validator1D = LayerValidator{ validator, INTEL_CONVOLUTIONAL_1D };
            return std::make_unique<const Target>(source,
                Validator{ validator1D, caps });
        }
        catch (const GnaException&)
        {
            // try 2D CNN in new arch
            return std::make_unique<const Target>(source,
                Validator{ validator, caps });
        }
    }

    static std::unique_ptr<const Component> CreateCnnComponent(const Shape& shape,
        const LayerValidator& validator, const FullCapabilitiesMap & caps)
    {
        if (shape.empty())
        {
            return CreateCnnTarget<Component, Shape>(
                Shape{ GNA_TENSOR_HW, 0u, 0u }, validator, caps);
        }
        else
        {
            return CreateCnnTarget<Component, Shape>(shape, validator, caps);
        }
    }

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
    template<class T>
    void InitOperationConfig(const T& operation)
    {
        OperationType = GetOperationType(operation);
        BiasesTensor = GetBiases(operation);
        if (BiasesTensor.Mode != Gna2TensorModeDisabled)
        {
            ModelErrorHelper::ExpectNotNull(BiasesTensor.Data, Gna2ItemTypeOperandData, BiasOperandIndex);
        }

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
    void InitPooling(const nn_layer& layer);

    void InitMultibias(const Gna2Operation& operation);
    void InitMultibias(const nn_layer& layer);

private:
    static Shape TryGetParamShape(const Gna2Operation & operation, OperationInfoKey parameter);
    static Shape TryGetParamShape(const Gna2Operation & operation, uint32_t parameterIndex);

    static const nn_layer_cnn2d* CastToCnn2DDetails(const nn_layer& layer);

    static Gna2Tensor GetWeights(const nn_layer& layer);
    static Gna2Tensor GetWeights(const Gna2Operation& operation);
    static Gna2Tensor GetFilters(const nn_layer& layer);
    static Gna2Tensor GetFilters(const Gna2Operation& operation);

    template<typename T>
    static Gna2Tensor getWeightsTensor(const nn_layer& layer, const T& affineFunc)
    {
        Gna2Tensor a{};
        a.Type = DataMode(affineFunc.nBytesPerWeight).Type;
        a.Data = affineFunc.pWeights;
        a.Mode = Gna2TensorModeDefault;
        if (layer.operation == INTEL_AFFINE_DIAGONAL)
        {
            a.Shape = { 1, layer.nOutputRows };
            a.Layout[0] = 'H';
            a.Layout[1] = '\0';
        }
        else if (layer.operation == INTEL_RECURRENT)
        {
            a.Shape = { 2, layer.nOutputColumns, layer.nInputColumns + layer.nOutputColumns };
            a.Layout[0] = 'H';
            a.Layout[1] = 'W';
            a.Layout[2] = '\0';
        }
        else
        {
            a.Shape = { 2, layer.nOutputRows, layer.nInputRows };
            a.Layout[0] = 'H';
            a.Layout[1] = 'W';
            a.Layout[2] = '\0';
        }

        return a;
    }

    template<typename T>
    static Gna2Tensor getBiasTensor(const nn_layer& layer, const T& affineFunc)
    {
        Gna2Tensor a{};
        a.Layout[0] = 'H';
        a.Layout[1] = '\0';
        a.Type = DataMode(affineFunc.nBytesPerBias).Type;
        a.Data = affineFunc.pBiases;
        a.Mode = Gna2TensorModeDefault;
        a.Shape = { 1, layer.operation == INTEL_RECURRENT
                        ? layer.nOutputColumns : layer.nOutputRows };

        return a;
    }

    static Gna2BiasMode GetBiasMode(const Gna2Operation& operation);
    static Gna2BiasMode GetBiasMode(const nn_layer& layer);
    static Gna2Tensor GetBiases(const Gna2Operation& operation);
    static Gna2Tensor GetBiases(const nn_layer& layer);

    static Shape GetStride(const Gna2Operation& operation);
    static Shape GetStride(const nn_layer& layer);

    static uint32_t GetFeedbackDelay(const Gna2Operation& operation);
    static uint32_t GetFeedbackDelay(const nn_layer& layer);

    static Shape GetZeroPadding(const Gna2Operation& operation);
    static Shape GetZeroPadding(const nn_layer& layer);

    static nn_layer_pool2d GetPoolingImpl(const nn_layer& layer);
    static Shape GetPoolingStride(const nn_layer_pool2d& pooling);
    static Shape GetPoolingWindow(const nn_layer_pool2d& pooling);
    static Shape GetPoolingWindow(const Gna2Operation& operation);
    static Shape GetPoolingStride(const Gna2Operation& operation);
    static Gna2PoolingMode GetPoolingMode(const nn_layer_pool2d& pooling);

    static bool hasPooling(const Gna2Operation& operation);
    static bool hasPooling(const nn_layer& layer);

    static bool isCNN2D(const nn_layer& layer);
    static bool isCNN2D(const Gna2Operation& operation);

    static bool isAffine(const nn_layer& layer);
    static bool isAffine(const Gna2Operation& operation);

    static bool isRecurrent(const nn_layer& layer);
    static bool isRecurrent(const Gna2Operation& operation);
};

}
