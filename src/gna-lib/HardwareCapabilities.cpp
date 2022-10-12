/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "HardwareCapabilities.h"

#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"

#include "gna2-capability-impl.h"

#include <memory>
#include <utility>

using namespace GNA;

// GNA hardware supports 256MB models, consisting of:
// - layer descriptors
// - user data
const uint32_t HardwareCapabilities::MaximumModelSize = 256 * 1024 * 1024;

template<Gna2DeviceVersion version>
static GenerationCapabilities GetVerCaps();

template<Gna2DeviceVersion baseVersion, Gna2DeviceGeneration targetGeneration>
static GenerationCapabilities DeriveVerCaps()
{
    static GenerationCapabilities caps = GetVerCaps<baseVersion>();
    caps.Generation = targetGeneration;
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionGMM>()
{
    return { Gna2DeviceGenerationGmm,
            1,
            { LegacyGMM },
            6,
            {{1, 8}},
            4,
            0,
            0,
            {0, 0, 0, 0, 0, 0, 0, 0,
                12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
            {},
    };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion0_9>()
{
    return {
    Gna2DeviceGeneration0_9,
    1023,
    { BaseFunctionality, LegacyGMM },
    6,
    {{2, 8}},
    4,
    1,
    1,
    {0, 0, 0, 0, 0, 0, 0, 0,
    12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
    {},
    };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion1_0>()
{
    static auto caps = DeriveVerCaps<Gna2DeviceVersion0_9, Gna2DeviceGeneration1_0>();
    caps.Features.insert(CNN);
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionEmbedded1_0>()
{
    static auto caps = DeriveVerCaps<Gna2DeviceVersion1_0, Gna2DeviceGeneration1_0>();
    caps.ComputeEngineCount = 3;
    caps.BufferElementCount =
    { 0, 0, 0, 0, 0, 0, 0, 0,
        6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144 };
    return caps;
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion2_0>()
{
    return  { Gna2DeviceGeneration2_0,
           4096,
           {
               BaseFunctionality, CNN, LegacyGMM, GMMLayer,
               MultiBias, NewPerformanceCounters
           },
           6,
           {{2, 8}},
           4,
           1,
           1,
           {0, 0, 0, 0, 0, 0, 0, 0,
               12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
           {},
            };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersion3_0>()
{
    return { Gna2DeviceGeneration3_0,
                8192,
                {
                     BaseFunctionality, CNN, LegacyGMM, GMMLayer,
                     MultiBias, NewPerformanceCounters, CNN2D, CNN1D
                },
                8,
                {{1, 16}, {2, 8}},
                4,
                2,
                16,
                {},
                {},
           };
}

template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionEmbedded3_1>()
{
    return DeriveVerCaps<Gna2DeviceVersion3_0, Gna2DeviceGeneration3_1>();
}

// for relaxed limits
// use DefaultDeviceVersion features as contains all known features to date
// no embedded as it should be only used for present device software mode
template<>
GenerationCapabilities GetVerCaps<Gna2DeviceVersionSoftwareEmulation>()
{
    static auto caps = DeriveVerCaps<Gna2DeviceVersion3_0, DefaultDeviceGeneration>();
    return caps;
}

template<Gna2DeviceVersion version>
static DevVerGenMap::allocator_type::value_type GetCaps()
{
    return { version, GetVerCaps<version>() };
}

DevVerGenMap& HardwareCapabilities::getCapsMap()
{
    static DevVerGenMap capsMap = {
         GetCaps<Gna2DeviceVersionGMM>(),
         GetCaps<Gna2DeviceVersion0_9>(),
         GetCaps<Gna2DeviceVersion1_0>(),
         GetCaps<Gna2DeviceVersionEmbedded1_0>(),
         GetCaps<Gna2DeviceVersion2_0>(),
         GetCaps<Gna2DeviceVersion3_0>(),
         GetCaps<Gna2DeviceVersionSoftwareEmulation>(),
         GetCaps<Gna2DeviceVersionEmbedded3_1>(),
    };

    // initialize remaining items that depend on capsMap values
    if (capsMap.at(Gna2DeviceVersion3_0).BufferElementCount[0] == 0)
    {
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersion3_0)));
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersion3_0)), true);
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionSoftwareEmulation)));
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionSoftwareEmulation)), true);
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionEmbedded3_1)));
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionEmbedded3_1)), true);
    }

    return capsMap;
}

const GenerationCapabilities&
HardwareCapabilities::getGenerationCapabilities(DeviceVersion deviceVersionIn)
{
    try
    {
        return getCapsMap().at(deviceVersionIn);
    }
    catch (std::out_of_range&)
    {
        throw GnaException(Gna2StatusDeviceVersionInvalid);
    }
}

bool HardwareCapabilities::Is3_0Generation(Gna2DeviceGeneration generation)
{
    return Gna2DeviceGeneration3_0 == generation || Gna2DeviceGeneration3_1 == generation;
}

bool HardwareCapabilities::Is3_0Device(DeviceVersion deviceVersionIn)
{
    auto const caps = getGenerationCapabilities(deviceVersionIn);
    return Is3_0Generation(caps.Generation);
}

uint32_t HardwareCapabilities::GetMaximumLayerCount(DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).MaximumLayerCount;
}

uint32_t HardwareCapabilities::GetComputeEngineCount(DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).ComputeEngineCount;
}

uint32_t HardwareCapabilities::GetBufferElementCount(
    DeviceVersion deviceVersionIn, uint32_t grouping, uint32_t inputPrecision)
{
    auto const index = ((inputPrecision - 1) * BufferArraySizeSingle) + grouping - 1;
    return getGenerationCapabilities(deviceVersionIn).BufferElementCount[index];
}

HardwareCapabilities::HardwareCapabilities(
    DeviceVersion deviceVersionIn) :
    deviceVersion{deviceVersionIn}
{
}

uint32_t const * HardwareCapabilities::GetHardwareConsistencySettings(
    DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).BufferElementCount.data();
}

uint32_t const * HardwareCapabilities::GetHardwareConsistencySettingsFor3_0(DeviceVersion deviceVersionIn)
{
    if (Is3_0Device(deviceVersionIn))
    {
        return getGenerationCapabilities(deviceVersionIn).BufferElementCount3_0Workaround.data();
    }
    return nullptr;
}

DeviceVersion HardwareCapabilities::GetDeviceVersion() const
{
    return deviceVersion;
}

Gna2DeviceGeneration HardwareCapabilities::GetDeviceGeneration(DeviceVersion deviceVersionIn)
{
    return getGenerationCapabilities(deviceVersionIn).Generation;
}

Gna2DeviceGeneration HardwareCapabilities::GetDeviceGeneration() const
{
    return GetDeviceGeneration(deviceVersion);
}

bool HardwareCapabilities::Is3_0Generation() const
{
    auto const generation = GetDeviceGeneration();
    return Is3_0Generation(generation);
}

bool HardwareCapabilities::IsOperationSupported(nn_operation operation) const
{
    static const std::map<nn_operation, GnaFeature> featureMap =
    {
        {INTEL_AFFINE, BaseFunctionality},
        {INTEL_AFFINE_DIAGONAL, BaseFunctionality},
        {INTEL_COPY, BaseFunctionality},
        {INTEL_DEINTERLEAVE, BaseFunctionality},
        {INTEL_INTERLEAVE, BaseFunctionality},
        {INTEL_RECURRENT, BaseFunctionality},
        {INTEL_AFFINE_MULTIBIAS, MultiBias},
        {INTEL_CONVOLUTIONAL, CNN},
        {INTEL_GMM, GMMLayer},
        {INTEL_CONVOLUTIONAL_2D, CNN2D},
        {INTEL_CONVOLUTIONAL_1D, CNN2D}, // mapped to 2D as INTEL_CONVOLUTIONAL_1D is only used to select limits here
    };

    return HasFeature(featureMap.at(operation));
}

bool HardwareCapabilities::HasFeature(GnaFeature feature) const
{
    const auto& caps = getGenerationCapabilities(deviceVersion);
    return contains(caps.Features, feature);
}

uint32_t HardwareCapabilities::GetMaximumLayerCount() const
{
    return GetMaximumLayerCount(deviceVersion);
}

void HardwareCapabilities::ValidateOperationCount(uint32_t operationCount) const
{
    ValidateOperationCount(operationCount, GetDeviceVersion());
}

void HardwareCapabilities::ValidateOperationCount(uint32_t operationCount, Gna2DeviceVersion version)
{
    ModelErrorHelper::ExpectGtZero(operationCount, Gna2ItemTypeModelNumberOfOperations);
    ModelErrorHelper::ExpectBelowEq(operationCount, GetMaximumLayerCount(version), Gna2ItemTypeModelNumberOfOperations);
}

void HardwareCapabilities::initHardwareConsistencySettings3_0(GenerationCapabilities& caps, bool isWorkaround)
{
    auto& buffers = isWorkaround ? caps.BufferElementCount3_0Workaround : caps.BufferElementCount;
    for (uint32_t p = 0; p < 2; p++)
    {
        auto const inputPrecision = isWorkaround ? 2 : p + 1;
        for (uint32_t i = 0; i < BufferArraySizeSingle; i++)
        {
            buffers[p * BufferArraySizeSingle + i] =
                getBufferElementCount3_0(caps.ComputeEngineCount, caps.BufferSizesPerCEInKB, i + 1, inputPrecision);
        }
    }
}

uint32_t HardwareCapabilities::getBufferElementCount3_0(uint32_t ceCount, uint32_t bufferSizeInKB,
    uint32_t grouping, uint32_t inputPrecision)
{
    auto count = (bufferSizeInKB * 1024)
        / (16 * grouping);
    count *= ceCount * 16 / inputPrecision;
    count *= grouping;

    return count;
}

uint32_t HardwareCapabilities::GetBufferSizeInKB(DeviceVersion deviceVersionIn)
{
    auto const& caps = getGenerationCapabilities(deviceVersionIn);
    return caps.ComputeEngineCount * caps.BufferSizesPerCEInKB;
}

HardwareCapabilitiesDevice::HardwareCapabilitiesDevice(const DriverCapabilities& discoveredDevice) :
    HardwareCapabilities{},
    isHardwareSupported{ isHwValid(discoveredDevice) },
    isSoftwareFallbackSupported{ isHardwareSupported && discoveredDevice.isSoftwareFallbackSupported }
{
    if (isHardwareSupported)
    {
        deviceVersion = discoveredDevice.deviceVersion;
    }
    // else Gna2DeviceVersionSoftwareEmulation is used
}

bool HardwareCapabilitiesDevice::isHwValid(const DriverCapabilities& discoveredDevice)
{
    if (Gna2DeviceVersionSoftwareEmulation == discoveredDevice.deviceVersion ||
        !contains(getCapsMap(), discoveredDevice.deviceVersion) ||
        discoveredDevice.hwInBuffSize != GetBufferSizeInKB(discoveredDevice.deviceVersion))
    {
        Log->Message("No compatible hardware detected.\n");
        return false;
    }
    return true;
}

HardwareCapabilitiesExport::HardwareCapabilitiesExport(DeviceVersion deviceVersionIn) :
    HardwareCapabilities{deviceVersionIn}
{
}
