/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "gna2-model-impl.h"
#include "gna2-common-impl.h"

#include "ApiWrapper.h"
#include "Device.h"
#include "DeviceManager.h"
#include "ModelError.h"
#include "ModelWrapper.h"
#include "StringHelper.h"

#include "gna2-model-api.h"
#include "gna2-common-api.h"

#include <stdint.h>

using namespace GNA;

GNA2_API enum Gna2Status Gna2ModelCreate(uint32_t deviceIndex,
    struct Gna2Model const * model, uint32_t * modelId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(model);
        Expect::NotNull(modelId);
        auto& device = DeviceManager::Get().GetDevice(deviceIndex);
        *modelId = device.LoadModel(*model);
        return Gna2StatusSuccess;
    };
    return ModelErrorHelper::ExecuteSafelyAndStoreLastError(command);
}

GNA2_API enum Gna2Status Gna2ModelRelease(uint32_t modelId)
{
    const std::function<ApiStatus()> command = [&]()
    {
        auto& device = DeviceManager::Get().GetDeviceForModel(modelId);
        device.ReleaseModel(modelId);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelGetLastError(struct Gna2ModelError * error)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(error);
        ModelErrorHelper::PopLastError(*error);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2ModelErrorGetMessage(
    struct Gna2ModelError const * const error,
    char * messageBuffer,
    uint32_t messageBufferSize)
{
    const std::function<ApiStatus()> command = [&]()
    {
        GNA::Expect::NotNull(messageBuffer);
        GNA::Expect::NotNull(error);
        const auto& message = ModelErrorHelper::GetErrorString(*error);
        GNA::StringHelper::Copy(*messageBuffer, messageBufferSize, message);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2ModelErrorGetMaxMessageLength()
{
    return ModelErrorHelper::GetErrorStringMaxLength();
}

GNA2_API enum Gna2Status Gna2ErrorTypeGetMessage(
    enum Gna2ErrorType type,
    char * messageBuffer,
    uint32_t messageBufferSize)
{
    const std::function<Gna2Status()> command = [&]()
    {
        GNA::Expect::NotNull(messageBuffer);
        const auto message = GNA::StringHelper::GetFromMap(ModelErrorHelper::GetAllErrorTypeStrings(), type);
        GNA::StringHelper::Copy(*messageBuffer, messageBufferSize, message);
        return Gna2StatusSuccess;
    };
    return GNA::ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2ErrorTypeGetMaxMessageLength()
{
    return StringHelper::GetMaxLength(ModelErrorHelper::GetAllErrorTypeStrings());
}

GNA2_API enum Gna2Status Gna2ItemTypeGetMessage(
    enum Gna2ItemType type,
    char * messageBuffer,
    uint32_t messageBufferSize)
{
    const std::function<Gna2Status()> command = [&]()
    {
        GNA::Expect::NotNull(messageBuffer);
        const auto message = GNA::StringHelper::GetFromMap(ModelErrorHelper::GetAllItemTypeStrings(), type);
        GNA::StringHelper::Copy(*messageBuffer, messageBufferSize, message);
        return Gna2StatusSuccess;
    };
    return GNA::ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2ItemTypeGetMaxMessageLength()
{
    return StringHelper::GetMaxLength(ModelErrorHelper::GetAllItemTypeStrings());
}

GNA2_API enum Gna2Status Gna2ModelOperationInit(struct Gna2Operation * operation,
    enum Gna2OperationType type, Gna2UserAllocator userAllocator)
{
    const std::function<ApiStatus()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, type, userAllocator);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2DataTypeGetSize(enum Gna2DataType type)
{
    const std::function<uint32_t()> command = [&]()
    {
        return ModelWrapper::DataTypeGetSize(type);
    };
    return ApiWrapper::ExecuteSafely(command, Gna2NotSupportedU32);
}

GNA2_API uint32_t Gna2ShapeGetNumberOfElements(struct Gna2Shape const * shape)
{
    const std::function<uint32_t()> command = [&]()
    {
        return ModelWrapper::ShapeGetNumberOfElements(shape);
    };
    return ApiWrapper::ExecuteSafely(command, Gna2NotSupportedU32);
}

GNA2_API uint32_t Gna2TensorGetSize(struct Gna2Tensor const * tensor)
{
    const std::function<uint32_t()> command = [&]()
    {
        Expect::NotNull(tensor);
        auto const apiTensor = std::make_unique<Tensor>(*tensor);
        return apiTensor->Size;
    };
    return ApiWrapper::ExecuteSafely(command, Gna2NotSupportedU32);
}

GNA2_API struct Gna2Shape Gna2ShapeInitScalar()
{
    const std::function<ApiShape()> command = []()
    {
        return ModelWrapper::ShapeInit();
    };
    return ApiWrapper::ExecuteSafely(command, Gna2Shape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit1D(uint32_t x)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(x);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit2D(uint32_t x, uint32_t y)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(x, y);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit3D(uint32_t x, uint32_t y, uint32_t z)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(x, y, z);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(n, x, y, z);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Shape Gna2ShapeInit6D(
        uint32_t d1, uint32_t d2, uint32_t d3,
        uint32_t d4, uint32_t d5, uint32_t d6)
{
    const std::function<ApiShape()> command = [&]()
    {
        return ModelWrapper::ShapeInit(d1, d2, d3, d4, d5, d6);
    };
    return ApiWrapper::ExecuteSafely(command, ApiShape{});
}

GNA2_API struct Gna2Tensor Gna2TensorInitDisabled()
{
    const std::function<ApiTensor()> command = [&]()
    {
        ApiTensor tensor {};
        tensor.Mode = Gna2TensorModeDisabled;
        return tensor;
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInitScalar(enum Gna2DataType type, void * data)
{
    const std::function<ApiTensor()> command = [&]()
    {
        auto tensor = ModelWrapper::TensorInit(type, {}, data );
        tensor.Layout[0] = 'S';
        return tensor;
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit1D(uint32_t x, enum Gna2DataType type,
    void * data)
{
    const std::function<ApiTensor()> command = [&]()
    {
        return ModelWrapper::TensorInit(type, {}, data, x );
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit2D(uint32_t x, uint32_t y,
    enum Gna2DataType type, void * data)
{
    const std::function<ApiTensor()> command = [&]()
    {
        return ModelWrapper::TensorInit(type, {}, data, x, y );
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit3D(uint32_t x, uint32_t y, uint32_t z,
    enum Gna2DataType type, void * data)
{
    const std::function<ApiTensor()> command = [&]()
    {
        return ModelWrapper::TensorInit(type, {}, data, x, y, z );
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInit4D(uint32_t n, uint32_t x, uint32_t y,
    uint32_t z, enum Gna2DataType type, void * data)
{
    const std::function<ApiTensor()> command = [&]()
    {
        return ModelWrapper::TensorInit(type, {}, data, n, x, y, z );
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API struct Gna2Tensor Gna2TensorInitActivation(uint32_t numberOfSegments,
    struct Gna2PwlSegment * segments)
{
    const std::function<ApiTensor()> command = [&]()
    {
        return ModelWrapper::TensorInit(Gna2DataTypePwlSegment, {},
                static_cast<void const *>(segments), numberOfSegments);
    };
    return ApiWrapper::ExecuteSafely(command, ApiTensor{});
}

GNA2_API enum Gna2Status Gna2OperationInitFullyConnectedAffine(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeFullyConnectedAffine, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, weights, biases, activation);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitElementWiseAffine(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeElementWiseAffine, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, weights, biases, activation);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitFullyConnectedBiasGrouping(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Tensor * weightScaleFactors,
    enum Gna2BiasMode * biasMode,
    uint32_t * biasVectorIndex)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeFullyConnectedAffine, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, weights, biases, activation, weightScaleFactors);

        Expect::NotNull(biasMode);
        ModelWrapper::SetParameters(*operation, biasMode, biasVectorIndex);

        *biasMode = Gna2BiasModeGrouping;
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitRecurrent(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * weights, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    uint32_t * delay)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeRecurrent, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, weights, biases, activation);
        ModelWrapper::SetParameters(*operation, delay);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitConvolution(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * convolutionStride,
    enum Gna2BiasMode * biasMode)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeConvolution, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, filters, biases, activation);
        ModelWrapper::SetParameters(*operation, convolutionStride, biasMode);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitConvolutionFused(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * filters, struct Gna2Tensor * biases,
    struct Gna2Tensor * activation,
    struct Gna2Shape * convolutionStride,
    enum Gna2BiasMode * biasMode,
    enum Gna2PoolingMode * poolingMode,
    struct Gna2Shape * poolingWindow,
    struct Gna2Shape * poolingStride,
    struct Gna2Shape * zeroPadding)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeConvolution, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, filters, biases, activation);
        ModelWrapper::SetParameters(*operation,
            convolutionStride, biasMode, poolingMode, poolingWindow, poolingStride, zeroPadding);

        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitCopy(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Shape * copyShape)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeCopy, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs);
        ModelWrapper::SetParameters(*operation, copyShape);

        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitTransposition(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeTransposition, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs);
        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitGmm(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * means,
    struct Gna2Tensor * inverseCovariances,
    struct Gna2Tensor * consts,
    uint32_t * maximumScore)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        Expect::NotNull(means);
        Expect::NotNull(inverseCovariances);
        Expect::NotNull(consts);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeGmm, userAllocator);
        ModelWrapper::SetOperands(*operation, inputs, outputs, means, inverseCovariances, consts);
        ModelWrapper::SetParameters(*operation, maximumScore);
        ModelWrapper::SetLayout(*inputs, "");
        ModelWrapper::SetLayout(*outputs, "");

        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}

GNA2_API enum Gna2Status Gna2OperationInitGmmInterleaved(
    struct Gna2Operation * operation, Gna2UserAllocator userAllocator,
    struct Gna2Tensor * inputs, struct Gna2Tensor * outputs,
    struct Gna2Tensor * interleavedTensors,
    uint32_t * maximumScore)
{
    const std::function<Gna2Status()> command = [&]()
    {
        Expect::NotNull(operation);
        ModelWrapper::OperationInit(*operation, Gna2OperationTypeGmm, userAllocator, true);
        ModelWrapper::SetOperands(*operation, inputs, outputs, interleavedTensors);
        ModelWrapper::SetParameters(*operation, maximumScore);
        ModelWrapper::SetLayout(*interleavedTensors, "HCWCWC");

        return Gna2StatusSuccess;
    };
    return ApiWrapper::ExecuteSafely(command);
}
