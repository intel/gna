/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "common.h"
#include "gna-api.h"
#include "gna-api-types-xnn.h"

#include <array>
#include <cstdint>
#include <map>

#include "gna2-common-impl.h"

namespace GNA
{
struct DriverCapabilities;

enum GnaFeature
{
    BaseFunctionality = 0, // DNN, DNN_AL, DIAGONAL, RNN, COPY, TRANSPOSE, PWL
    CNN,
    LegacyGMM,
    GMMLayer,
    MultiBias,
    L1Distance,
    L2Distance,
    ComputerVision,
    NewPerformanceCounters,
    CNN2D,
};

// buffer array size for single precision
static constexpr uint32_t BufferArraySizeSingle = XNN_N_GROUP_MAX;
static constexpr uint32_t BufferArraySize = 2 * BufferArraySizeSingle;

struct GenerationCapabilities
{
    gna_device_generation Generation;
    uint32_t MaximumLayerCount;
    std::map<GnaFeature, bool> Features;
    uint32_t ComputeEngineCount;
    std::map<const uint32_t /* input precision */, const uint32_t> MacCountPerCE;
    uint32_t BufferSizesPerCEInKB;
    uint32_t PoolingEngineCountPerCE;
    uint32_t ActivationEngineCount;
    std::array<uint32_t, BufferArraySize> BufferElementCount;
    std::array<uint32_t, BufferArraySize> BufferElementCount3_0Workaround;
};

class HardwareCapabilities
{
public:
    explicit HardwareCapabilities(DeviceVersion deviceVersionIn = DefaultDeviceVersion);

    void DiscoverHardware(const DriverCapabilities& discoveredDriver);

    static uint32_t const * GetHardwareConsistencySettings(DeviceVersion deviceVersion);
    static uint32_t const * GetHardwareConsistencySettingsFor3_0(DeviceVersion deviceVersion);

    // For now all hardware generations share the same maximum model size
    // in the future it's possible to integrate it as GenerationCapabilities field
    static const uint32_t MaximumModelSize;

    static bool Is3_0Generation(gna_device_generation generation);
    static bool Is3_0Device(DeviceVersion deviceVersion);

    static DeviceVersion GetDeviceVersion(gna_device_generation generation);

    static uint32_t GetMaximumLayerCount(DeviceVersion deviceVersion);

    static uint32_t GetComputeEngineCount(DeviceVersion deviceVersion);

    // Gets the number of data elements that may be stored in hw buffer
    static uint32_t GetBufferElementCount(DeviceVersion deviceVersion,
        uint32_t grouping, uint32_t inputPrecision = GNA_INT16);

    uint32_t GetBufferElementCount(uint32_t grouping, uint32_t inputPrecision = GNA_INT16) const
    {
        return GetBufferElementCount(deviceVersion, grouping, inputPrecision);
    }

    DeviceVersion GetDeviceVersion() const;

    DeviceVersion GetHardwareDeviceVersion() const
    {
        return IsHardwareSupported()
            ? GetDeviceVersion()
            : Gna2DeviceVersionSoftwareEmulation;
    }

    gna_device_generation GetDeviceGeneration() const;

    bool Is3_0Generation() const;

    uint32_t GetMaximumLayerCount() const;

    bool IsLayerSupported(nn_operation operation) const;

    bool IsHardwareSupported() const
    {
        return hardwareSupported;
    }

    bool HasFeature(GnaFeature feature) const;

private:
    static std::map<DeviceVersion, const GenerationCapabilities>& getCapsMap();

    static const GenerationCapabilities& getGenerationCapabilities(DeviceVersion deviceVersionIn);

    static void initHardwareConsistencySettings3_0(GenerationCapabilities& caps, bool isWorkaround = false);

    static uint32_t getBufferElementCount3_0(uint32_t ceCount, uint32_t bufferSizeInKB,
        uint32_t grouping, uint32_t inputPrecision = GNA_INT16);

    uint32_t GetBufferSizeInKB() const;

    static uint32_t GetBufferSizeInKB(DeviceVersion deviceVersion);

    bool hardwareSupported = false;

    DeviceVersion deviceVersion;

    uint32_t bufferSize;

    uint32_t driverRecoveryTimeout = 0;
};

}
