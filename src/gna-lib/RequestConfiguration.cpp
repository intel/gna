/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "RequestConfiguration.h"

#include "ActiveList.h"
#include "Address.h"
#include "CompiledModel.h"
#include "Expect.h"
#include "HardwareCapabilities.h"
#include "KernelArguments.h"
#include "Layer.h"
#include "LayerConfiguration.h"

#include "gna-api-status.h"

#include <memory>
#include <utility>

using namespace GNA;

RequestConfiguration::RequestConfiguration(CompiledModel& model, uint32_t configId,
    DeviceVersion consistentDeviceIn) :
    Model{ model },
    Id{ configId },
    BufferElementCount{ HardwareCapabilities::GetHardwareConsistencySettings(consistentDeviceIn) },
    consistentDevice{ consistentDeviceIn }
{
}

void RequestConfiguration::AddBuffer(uint32_t operandIndex, uint32_t layerIndex, void *address)
{
    auto context = AddBufferContext(Model, operandIndex, layerIndex, address);
    storeAllocationIfNew(context.Address, context.Size);

    if (ScratchpadOperandIndex == layerIndex)
    {
        addBufferForMultipleLayers(context);
    }
    else
    {
        addBufferForSingleLayer(context);

    }
    Model.InvalidateHardwareRequestConfig(Id);
}

RequestConfiguration::AddBufferContext::AddBufferContext(CompiledModel & model,
    uint32_t operandIndexIn, uint32_t layerIndexIn, void * addressIn) :
    SoftwareLayer{ nullptr },
    Operand{ nullptr },
    OperandIndex{ operandIndexIn },
    LayerIndex{ layerIndexIn },
    Address{ addressIn }
{
    Expect::NotNull(Address);

    if (ScratchpadOperandIndex == LayerIndex)
    {
        Size = model.GetMaximumOperandSize(OperandIndex);
    }
    else
    {
        SoftwareLayer = &model.GetLayer(LayerIndex);
        Operand = &SoftwareLayer->GetOperand(OperandIndex);
        Size = Operand->Size;
    }
}

void RequestConfiguration::addBufferForMultipleLayers(AddBufferContext & context)
{
    context.LayerIndex = 0;
    for (auto const & layerIter : Model.GetLayers())
    {
        context.SoftwareLayer = layerIter.get(); // not null assured
        context.Operand = context.SoftwareLayer->TryGetOperand(context.OperandIndex);

        applyBufferForSingleLayer(context);
        context.LayerIndex++;
    }
}

void RequestConfiguration::addBufferForSingleLayer(AddBufferContext & context)
{
    context.SoftwareLayer = &Model.GetLayer(context.LayerIndex);
    Expect::NotNull(context.Operand, Gna2StatusXnnErrorLyrCfg);
    applyBufferForSingleLayer(context);
}


void RequestConfiguration::applyBufferForSingleLayer(AddBufferContext & context)
{
    if (nullptr != context.Operand)
    {
        auto & layerConfiguration = getLayerConfiguration(context.LayerIndex);
        layerConfiguration.EmplaceBuffer(context.OperandIndex, context.Address);

        // if invalidate fails, we don't know if it's already been used thus no recovery from this
        context.SoftwareLayer->UpdateKernelConfigs(layerConfiguration);
    }
}

LayerConfiguration & RequestConfiguration::getLayerConfiguration(uint32_t layerIndex)
{
    auto const found = LayerConfigurations.emplace(layerIndex, std::make_unique<LayerConfiguration>());
    auto & layerConfiguration = *found.first->second;
    return layerConfiguration;
}

void RequestConfiguration::AddActiveList(uint32_t layerIndex, const ActiveList& activeList)
{
    const auto& layer = Model.GetLayer(layerIndex);

    Expect::InSet(layer.Operation, { INTEL_AFFINE, INTEL_GMM }, Gna2StatusXnnErrorLyrOperation);

    auto & layerConfiguration = getLayerConfiguration(layerIndex);

    Expect::Null(layerConfiguration.ActList.get());

    storeAllocationIfNew(activeList.Indices,
        activeList.IndicesCount * static_cast<uint32_t>(sizeof(uint32_t)));

    auto activeListPtr = ActiveList::Create(activeList);
    layerConfiguration.ActList.swap(activeListPtr);
    ++ActiveListCount;

    layer.UpdateKernelConfigs(layerConfiguration);
    Model.InvalidateHardwareRequestConfig(Id);
}

void RequestConfiguration::SetHardwareConsistency(
    DeviceVersion consistentDeviceIn)
{
    if (Gna2DeviceVersionSoftwareEmulation != consistentDeviceIn)
    {
        Expect::True(Model.IsFullyHardwareCompatible(HardwareCapabilities{ consistentDeviceIn }), Gna2StatusAccelerationModeNotSupported);
        BufferElementCount = HardwareCapabilities::GetHardwareConsistencySettings(consistentDeviceIn);
        BufferElementCountFor3_0 = HardwareCapabilities::GetHardwareConsistencySettingsFor3_0(consistentDeviceIn);
    }
    Acceleration.SetHwConsistency(Gna2DeviceVersionSoftwareEmulation != consistentDeviceIn);
    consistentDevice = consistentDeviceIn;
}

void RequestConfiguration::EnforceAcceleration(Gna2AccelerationMode accelMode)
{
    if (accelMode == Gna2AccelerationModeHardware)
    {
        Expect::True(Model.IsHardwareEnforcedModeValid(), Gna2StatusAccelerationModeNotSupported);
    }
    Acceleration.SetMode(accelMode);
}

DeviceVersion RequestConfiguration::GetConsistentDevice() const
{
    return consistentDevice;
}

uint8_t RequestConfiguration::GetHwInstrumentationMode() const
{
    if (profilerConfiguration != nullptr)
    {
        return profilerConfiguration->GetHwPerfEncoding();
    }
    return 0;
}

void RequestConfiguration::storeAllocationIfNew(void const *buffer, uint32_t bufferSize)
{
    // add buffer memory if is not already included in model memory
    auto const memory = Model.GetMemoryIfNotPartOfModel(buffer, bufferSize);
    if (nullptr != memory)
    {
        Model.ValidateBuffer(allocations, *memory);
        allocations.Emplace(*memory);
    }
    // else buffer already in model memory
}
