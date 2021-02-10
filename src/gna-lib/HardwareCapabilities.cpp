/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "HardwareCapabilities.h"

#include "DriverInterface.h"
#include "Expect.h"
#include "GnaException.h"
#include "Logger.h"
#include "Macros.h"

#include "gna-api-status.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

using namespace GNA;

// GNA hardware supports 256MB models, consisting of:
// - layer descriptors
// - user data
const uint32_t HardwareCapabilities::MaximumModelSize = 256 * 1024 * 1024;

std::map<DeviceVersion, const GenerationCapabilities>& HardwareCapabilities::getCapsMap()
{
    static std::map<DeviceVersion, const GenerationCapabilities> capsMap = {
        { Gna2DeviceVersionGMM,
            {GMM_DEVICE,
            1,
            {
                { BaseFunctionality, false},
                { CNN, false },
                { LegacyGMM, true },
                { GMMLayer, false },
                { MultiBias, false },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, false },
                { CNN2D, false }
            },
            6,
            {{1, 8}},
            4,
            0,
            0,
            {0, 0, 0, 0, 0, 0, 0, 0,
                12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
            {}},
        },
        { Gna2DeviceVersion0_9,
            {GNA_0_9,
            1023,
            {
                { BaseFunctionality, true},
                { CNN, false },
                { LegacyGMM, true },
                { GMMLayer, false },
                { MultiBias, false },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, false },
                { CNN2D, false }
            },
            6,
            {{2, 8}},
            4,
            1,
            1,
            {0, 0, 0, 0, 0, 0, 0, 0,
                12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
            {}},
        },
        { Gna2DeviceVersion1_0,
            {GNA_1_0,
            1023,
            {
                { BaseFunctionality, true},
                { CNN, true },
                { LegacyGMM, true },
                { GMMLayer, false },
                { MultiBias, false },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, false },
                { CNN2D, false }
            },
            6,
            {{2, 8}},
            4,
            1,
            1,
            {0, 0, 0, 0, 0, 0, 0, 0,
                12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
            {}},
        },
        { Gna2DeviceVersionEmbedded1_0,
            {GNA_1_0,
            1023,
            {
                { BaseFunctionality, true},
                { CNN, true },
                { LegacyGMM, true },
                { GMMLayer, false },
                { MultiBias, false },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, false },
                { CNN2D, false }
            },
            3,
            {{2, 8}},
            4,
            1,
            1,
            {0, 0, 0, 0, 0, 0, 0, 0,
                6144, 6144, 6048, 6144, 5760, 6048, 6048, 6144},
            {}},
        },
        { Gna2DeviceVersion2_0,
            {GNA_2_0,
            4096,
            {
                { BaseFunctionality, true},
                { CNN, true },
                { LegacyGMM, true },
                { GMMLayer, true },
                { MultiBias, true },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, true },
                { CNN2D, false }
            },
            6,
            {{2, 8}},
            4,
            1,
            1,
            {0, 0, 0, 0, 0, 0, 0, 0,
                12288, 12288, 12096, 12288, 12000, 12096, 12096, 12288},
            {}},
        },
            { Gna2DeviceVersionFromInt(0x30),
            {GNA_3_0,
            8191,
            {
                { BaseFunctionality, true},
                { CNN, true },
                { LegacyGMM, true },
                { GMMLayer, true },
                { MultiBias, true },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, true },
                { CNN2D, true }
            },
            8,
            {{1, 16}, {2, 8}},
            4,
            2,
            16,
            {},
            {}},
        },
        { Gna2DeviceVersionFromInt(0x30E),
            {GNA_3_0,
            8191,
            {
                { BaseFunctionality, true},
                { CNN, true },
                { LegacyGMM, true },
                { GMMLayer, true },
                { MultiBias, true },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, true },
                { CNN2D, true }
            },
            8,
            {{1, 16}, {2, 8}},
            4,
            2,
            16,
            {},
            {}},
        },
        { Gna2DeviceVersionFromInt(0x31E),
            {GNA_3_0,
            8191,
            {
                { BaseFunctionality, true},
                { CNN, true },
                { LegacyGMM, true },
                { GMMLayer, true },
                { MultiBias, true },
                { L1Distance, false },
                { L2Distance, false },
                { ComputerVision, false },
                { NewPerformanceCounters, true },
                { CNN2D, true }
            },
            2,
            {{1, 16}, {2, 8}},
            4,
            2,
            16,
            {},
            {}},
        },
    };

    // initialize remaining items that depend on capsMap values
    if (capsMap.at(Gna2DeviceVersionFromInt(0x30)).BufferElementCount[0] == 0)
    {
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionFromInt(0x30))));
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionFromInt(0x30))), true);
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionFromInt(0x30E))));
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionFromInt(0x30E))), true);
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionFromInt(0x31E))));
        initHardwareConsistencySettings3_0(const_cast<GenerationCapabilities&>(capsMap.at(Gna2DeviceVersionFromInt(0x31E))), true);
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

bool HardwareCapabilities::Is3_0Generation(gna_device_generation generation)
{
    return GNA_3_0 == generation;
}

bool HardwareCapabilities::Is3_0Device(DeviceVersion deviceVersionIn)
{
    auto const caps = getGenerationCapabilities(deviceVersionIn);
    return Is3_0Generation(caps.Generation);
}

DeviceVersion HardwareCapabilities::GetDeviceVersion(gna_device_generation generation)
{
    auto type = std::find_if(getCapsMap().cbegin(), getCapsMap().cend(),
        [generation](const std::pair<const DeviceVersion, const GenerationCapabilities>& genCaps)
    {
        return genCaps.second.Generation == generation;
    });
    return type->first;
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
    deviceVersion{deviceVersionIn},
    bufferSize{GetBufferSizeInKB()}
{
}

void HardwareCapabilities::DiscoverHardware(const DriverCapabilities& discoveredDriver)
{
    auto const hwInBuffSize = discoveredDriver.hwInBuffSize;
    if (discoveredDriver.deviceVersion == Gna2DeviceVersionFromInt(0x30) ||
        1 != getCapsMap().count(discoveredDriver.deviceVersion) ||
        hwInBuffSize != GetBufferSizeInKB(discoveredDriver.deviceVersion))
    {
        Log->Message("No compatible hardware detected.\n");
        return;
    }

    deviceVersion = discoveredDriver.deviceVersion;
    bufferSize = hwInBuffSize;
    driverRecoveryTimeout = discoveredDriver.recoveryTimeout;

    hardwareSupported = true;
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

gna_device_generation HardwareCapabilities::GetDeviceGeneration() const
{
    return getGenerationCapabilities(deviceVersion).Generation;
}

bool HardwareCapabilities::Is3_0Generation() const
{
    auto const generation = GetDeviceGeneration();
    return Is3_0Generation(generation);
}

bool HardwareCapabilities::IsLayerSupported(nn_operation operation) const
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
    };

    return HasFeature(featureMap.at(operation));
}

bool HardwareCapabilities::HasFeature(GnaFeature feature) const
{
    const auto& caps = getGenerationCapabilities(deviceVersion);
    return caps.Features.at(feature);
}

uint32_t HardwareCapabilities::GetMaximumLayerCount() const
{
    return GetMaximumLayerCount(deviceVersion);
}

void HardwareCapabilities::initHardwareConsistencySettings3_0(GenerationCapabilities& caps, bool isWorkaround)
{
    auto& buffers = (isWorkaround) ? caps.BufferElementCount3_0Workaround : caps.BufferElementCount;
    for (uint32_t p = 0; p < 2; p++)
    {
        auto const inputPrecision = (isWorkaround) ? 2 : p + 1;
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

uint32_t HardwareCapabilities::GetBufferSizeInKB() const
{
    return GetBufferSizeInKB(deviceVersion);
}
