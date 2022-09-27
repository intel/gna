/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Layer.h"

#include "AffineFunctions.h"
#include "AffineLayers.h"
#include "ConvolutionalLayer.h"
#include "ConvolutionalLayer2D.h"
#include "CopyLayer.h"
#include "DataMode.h"
#include "Expect.h"
#include "GmmLayer.h"
#include "LayerConfiguration.h"
#include "Logger.h"
#include "ModelError.h"
#include "ModelWrapper.h"
#include "RecurrentLayer.h"
#include "TransposeLayer.h"

#include <map>
#include <utility>

using namespace GNA;

std::unique_ptr<Layer> Layer::Create(const Gna2Operation & operation, const BaseValidator & validatorIn)
{
    ModelWrapper::ExpectOperationValid(operation);

    auto const defaultLayerValidator = LayerValidator{ validatorIn, toLegacy(operation, validatorIn) };

    switch (operation.Type)
    {
    case Gna2OperationTypeFullyConnectedAffine:
    {
        if (ModelWrapper::HasEnabledOperand(operation, WeightScaleFactorOperandIndex))
        {
            ModelWrapper::ExpectParameterAvailable(operation, BiasModeAffineParamIndex);
            ModelWrapper::ExpectParameterAvailable(operation, BiasVectorParamIndex);
        }
        return std::make_unique<AffineLayer>(operation, defaultLayerValidator);
    }
    case Gna2OperationTypeElementWiseAffine:
        return std::make_unique<AffineLayer>(operation, defaultLayerValidator);
    case Gna2OperationTypeRecurrent:
        return std::make_unique<RecurrentLayer>(operation, defaultLayerValidator);
    case Gna2OperationTypeCopy:
        return std::make_unique<CopyLayer>(operation, defaultLayerValidator);
    case Gna2OperationTypeConvolution:
    {
        if (CnnLayer::IsForced(operation))
        {
            Log->Message("Processing in Legacy CNN1D enforced mode.\n");
            return std::make_unique<CnnLayer>(operation, defaultLayerValidator);
        }
        Log->Message("Processing in New CNN2D mode.\n");
        try
        {
            // try new CNN using 1D variant
            auto const validator1D = LayerValidator{ validatorIn, INTEL_CONVOLUTIONAL_1D };
            return std::make_unique<ConvolutionalLayer2D>(operation, validator1D);
        }
        catch (const GnaException&)
        {
            // continue to 2D CNN
            auto const validator2D = LayerValidator{ validatorIn, INTEL_CONVOLUTIONAL_2D };
            return std::make_unique<ConvolutionalLayer2D>(operation, validator2D);
        }
    }
    case Gna2OperationTypeGmm:
        return std::make_unique<GmmOperation>(operation, defaultLayerValidator);
    case Gna2OperationTypeTransposition:
        return std::make_unique<TransposeLayer>(operation, defaultLayerValidator);
    default:
        throw GnaModelErrorException(
            Gna2ItemTypeOperationType,
            Gna2ErrorTypeNotInSet,
            operation.Type);
    }
}

void Layer::addBufferAs(const BufferMap& source, uint32_t sourceType,
    BufferMap& destination, uint32_t destinationType) const
{
    if (ScratchpadOperandIndex == sourceType && Transforms.size() < 2)
    {
        return;
    }

    const auto buffer = source.find(sourceType);
    if (buffer != source.end())
    {
        destination[destinationType] = buffer->second;
    }
}

void Layer::UpdateKernelConfigs(LayerConfiguration& layerConfiguration) const
{
    if (!Transforms.empty())
    {
        auto nonIoBuffers = layerConfiguration.Buffers;
        nonIoBuffers.erase(InputOperandIndex);
        nonIoBuffers.erase(OutputOperandIndex);
        nonIoBuffers.erase(ScratchpadOperandIndex);

        for (auto transform = Transforms.cbegin(); transform != Transforms.cend(); ++transform)
        {
            BufferMap buffers = nonIoBuffers;
            if (transform == Transforms.cbegin())
            {
                addBufferAs(layerConfiguration.Buffers, InputOperandIndex,
                    buffers, InputOperandIndex);
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, OutputOperandIndex);
            }
            if (transform == --Transforms.cend())
            {
                addBufferAs(layerConfiguration.Buffers, OutputOperandIndex,
                    buffers, OutputOperandIndex);
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, InputOperandIndex);
            }
            if (transform != Transforms.cbegin() && transform != --Transforms.cend())
            {
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, InputOperandIndex);
                addBufferAs(layerConfiguration.Buffers, ScratchpadOperandIndex,
                    buffers, OutputOperandIndex);
            }
            transform->get()->UpdateConfigBuffers(layerConfiguration.ConfigList, buffers);
        }

        if (layerConfiguration.ActList)
        {
            GetOutputTransform().ValidateActiveList(*layerConfiguration.ActList);
        }
    }
}

const DataConfig Layer::GetDataMode() const
{
    return dataConfig;
}

Tensor const & Layer::GetOperand(uint32_t operandIndex) const
{
    switch (operandIndex)
    {
    case InputOperandIndex:
        return Input;
    case OutputOperandIndex:
        return Output;
    default:
        throw GnaException(Gna2StatusXnnErrorLyrCfg);
    }
}

Tensor const * Layer::TryGetOperand(uint32_t operandIndex) const
{
    try
    {
        return &GetOperand(operandIndex);
    }
    catch (const GnaException&)
    {
        return nullptr;
    }
}

uint32_t Layer::TryGetOperandSize(uint32_t operandIndex) const
{
    auto const operand = TryGetOperand(operandIndex);
    if (nullptr != operand)
    {
        return operand->Size;
    }
    return 0;
}

void Layer::VerifyHas1BInputAnd2BWeight()
{
    if (is1BInputAnd2BWeightVerified)
    {
        return;
    }

    is1BInputAnd2BWeightVerified = true;

    auto const input = TryGetOperand(InputOperandIndex);
    auto const weight = TryGetOperand(WeightOperandIndex);
    if (input &&
        weight &&
        Gna2DataTypeInt8 == input->Mode.Type &&
        Gna2DataTypeInt16 == weight->Mode.Type)
    {
        has1BInputAnd2BWeight = true;
    }
}

Tensor const & Layer::getTransformOperand(TransformOperation operation, uint32_t operandIndex) const
{
    auto const & transform = Transforms.Get(operation);
    return transform.GetOperand(operandIndex);
}

void Layer::initTransforms(const std::vector<TransformOperation>& transforms,
    TransformFactoryConfig & commonConfig, const OperationConfig & operationConfig)
{
    for (const auto& transform : transforms)
    {
        outputTransform = Transforms.Emplace(transform, commonConfig, operationConfig);
        commonConfig.input = outputTransform->Output.get();
    }
    Expect::NotNull(outputTransform);

    inputTransform = Transforms.begin()->get();
    Expect::NotNull(inputTransform);

    if (Output.Buffer)
    {
        outputTransform->SetOutput(Output.Buffer);
    }

    if (transforms.back() == ActivationTransform
        && outputTransform->Operation != ActivationTransform)
    {
        const auto outType = outputTransform->Output->Mode.Type;
        auto const ctx = ModelItem{ Gna2ItemTypeOperandType, OutputOperandIndex };
        ModelErrorHelper::ExpectInSet(outType, { Gna2DataTypeInt32 }, ctx);
    }
}

void Layer::initComputeFunctions()
{
    ComputeHidden = [this](AccelerationMode accel, ExecutionConfig const & executionConfig)
    {this->compute(nullptr, accel, executionConfig); };

    Compute = [this](LayerConfiguration &layerConfiguration,
        AccelerationMode accel,
        ExecutionConfig const & executionConfig)
    {this->compute(&layerConfiguration, accel, executionConfig); };
}

void Layer::compute(const LayerConfiguration* layerConfiguration, AccelerationMode accel,
    ExecutionConfig const& execution) const
{
    for (const auto& transform : Transforms)
    {
        if (transform)
        {
            transform->Compute(accel, layerConfiguration, execution);
        }
    }
}

nn_operation AbstractOperation::toLegacy(
    const Gna2Operation& operation, const BaseValidator& validator)
{
    switch (operation.Type)
    {
    case Gna2OperationTypeElementWiseAffine:
        return INTEL_AFFINE_DIAGONAL;
    case Gna2OperationTypeFullyConnectedAffine:
        if (OperationConfig::IsMultibias(operation))
        {
            return INTEL_AFFINE_MULTIBIAS;
        }
        return INTEL_AFFINE;
    case Gna2OperationTypeCopy:
        return INTEL_COPY;
    case Gna2OperationTypeTransposition:
    {
        const Gna2Tensor& inputTensor = *operation.Operands[InputOperandIndex];
        if (LayerInput::IsInputInterleave(inputTensor, validator))
        {
            return INTEL_INTERLEAVE;
        }
        return INTEL_DEINTERLEAVE;
    }
    case Gna2OperationTypeRecurrent:
        return INTEL_RECURRENT;
    case Gna2OperationTypeConvolution:
        if(CnnLayer::IsForced(operation))
        {
            return INTEL_CONVOLUTIONAL;
        }
        return INTEL_CONVOLUTIONAL_2D;
    case Gna2OperationTypeGmm:
        return INTEL_GMM;
    default:
        throw GnaException(Gna2StatusNotImplemented);
    }
}

Gna2OperationType AbstractOperation::fromLegacy(const nn_operation& layerType)
{
    static const std::map<nn_operation, Gna2OperationType> operationTypes =
    {
        {INTEL_AFFINE, Gna2OperationTypeFullyConnectedAffine},
        {INTEL_AFFINE_DIAGONAL, Gna2OperationTypeElementWiseAffine},
        {INTEL_AFFINE_MULTIBIAS, Gna2OperationTypeFullyConnectedAffine},
        {INTEL_CONVOLUTIONAL, Gna2OperationTypeConvolution},
        {INTEL_CONVOLUTIONAL_1D, Gna2OperationTypeConvolution},
        {INTEL_CONVOLUTIONAL_2D, Gna2OperationTypeConvolution},
        {INTEL_COPY, Gna2OperationTypeCopy},
        {INTEL_DEINTERLEAVE, Gna2OperationTypeTransposition},
        {INTEL_GMM, Gna2OperationTypeGmm},
        {INTEL_INTERLEAVE, Gna2OperationTypeTransposition},
        {INTEL_RECURRENT, Gna2OperationTypeRecurrent},
    };

    try
    {
        return operationTypes.at(layerType);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

