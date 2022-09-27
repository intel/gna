/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Capabilities.h"

#include "WindowsDriverInterface.h"

#include "gna2-capability-api.h"
#include "gna2-common-impl.h"

#include <array>
#include <cstdint>
#include <map>
#include <set>

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
    NewPerformanceCounters,
    CNN2D,
    CNN1D, // Only used for switching HW LD mode
};

// buffer array size for single precision
static constexpr uint32_t BufferArraySizeSingle = BatchSizeMax;
static constexpr uint32_t BufferArraySize = 2 * BufferArraySizeSingle;

struct GenerationCapabilities
{
    Gna2DeviceGeneration Generation;
    uint32_t MaximumLayerCount;
    std::set<GnaFeature> Features;
    uint32_t ComputeEngineCount;
    std::map<const uint32_t /* input precision */, const uint32_t> MacCountPerCE;
    uint32_t BufferSizesPerCEInKB;
    uint32_t PoolingEngineCountPerCE;
    uint32_t ActivationEngineCount;
    std::array<uint32_t, BufferArraySize> BufferElementCount;
    std::array<uint32_t, BufferArraySize> BufferElementCount3_0Workaround;
};

using DevVerGenMap = std::map<DeviceVersion, const GenerationCapabilities>;

class HardwareCapabilities
{
public:
    virtual ~HardwareCapabilities() = default;
    HardwareCapabilities(const HardwareCapabilities&) = delete;
    HardwareCapabilities(HardwareCapabilities&&) = delete;
    HardwareCapabilities& operator=(const HardwareCapabilities&) = delete;
    HardwareCapabilities& operator=(HardwareCapabilities&&) = delete;

    static uint32_t const * GetHardwareConsistencySettings(DeviceVersion deviceVersionIn);
    static uint32_t const * GetHardwareConsistencySettingsFor3_0(DeviceVersion deviceVersionIn);

    // For now all hardware generations share the same maximum model size
    // in the future it's possible to integrate it as GenerationCapabilities field
    static const uint32_t MaximumModelSize;

    static bool Is3_0Device(DeviceVersion deviceVersionIn);

    static uint32_t GetMaximumLayerCount(DeviceVersion deviceVersionIn);

    static uint32_t GetComputeEngineCount(DeviceVersion deviceVersionIn);

    uint32_t GetBufferElementCount(uint32_t grouping, uint32_t inputPrecision = Gna2DataTypeInt16) const
    {
        return GetBufferElementCount(deviceVersion, grouping, inputPrecision);
    }

    DeviceVersion GetDeviceVersion() const;

    virtual DeviceVersion GetHardwareDeviceVersion() const
    {
        return GetDeviceVersion();
    }

    static Gna2DeviceGeneration GetDeviceGeneration(DeviceVersion deviceVersionIn);

    Gna2DeviceGeneration GetDeviceGeneration() const;

    bool Is3_0Generation() const;

    uint32_t GetMaximumLayerCount() const;

    void ValidateOperationCount(uint32_t operationCount) const;

    static void ValidateOperationCount(uint32_t operationCount, Gna2DeviceVersion version);

    bool IsOperationSupported(nn_operation operation) const;

    virtual bool IsHardwareSupported() const
    {
        return false;
    }

    bool HasFeature(GnaFeature feature) const;

    virtual bool IsSoftwareFallbackSupported() const
    {
        return false;
    }

protected:
    explicit HardwareCapabilities(DeviceVersion deviceVersionIn = DefaultDeviceVersion);

    static bool Is3_0Generation(Gna2DeviceGeneration generation);

    static DevVerGenMap& getCapsMap();

    static const GenerationCapabilities& getGenerationCapabilities(DeviceVersion deviceVersionIn);

    static void initHardwareConsistencySettings3_0(GenerationCapabilities& caps, bool isWorkaround = false);

    // Gets the number of data elements that may be stored in hw buffer
    static uint32_t GetBufferElementCount(DeviceVersion deviceVersionIn,
        uint32_t grouping, uint32_t inputPrecision = Gna2DataTypeInt16);

    static uint32_t getBufferElementCount3_0(uint32_t ceCount, uint32_t bufferSizeInKB,
        uint32_t grouping, uint32_t inputPrecision = Gna2DataTypeInt16);

    static uint32_t GetBufferSizeInKB(DeviceVersion deviceVersionIn);

    DeviceVersion deviceVersion;
};

class HardwareCapabilitiesDevice final : public HardwareCapabilities
{
public:
    explicit HardwareCapabilitiesDevice(const DriverCapabilities& discoveredDevice);
    virtual ~HardwareCapabilitiesDevice() = default;

    HardwareCapabilitiesDevice(const HardwareCapabilitiesDevice&) = delete;
    HardwareCapabilitiesDevice(HardwareCapabilitiesDevice&&) = delete;
    HardwareCapabilitiesDevice& operator=(const HardwareCapabilitiesDevice&) = delete;
    HardwareCapabilitiesDevice& operator=(HardwareCapabilitiesDevice&&) = delete;

    DeviceVersion GetHardwareDeviceVersion() const override
    {
        return IsHardwareSupported()
            ? GetDeviceVersion()
            : Gna2DeviceVersionSoftwareEmulation;
    }

    bool IsHardwareSupported() const override
    {
        return isHardwareSupported;
    }

    bool IsSoftwareFallbackSupported() const override
    {
        return isSoftwareFallbackSupported;
    }

protected:
    static bool isHwValid(const DriverCapabilities& discoveredDevice);

    const bool isHardwareSupported;

    const bool isSoftwareFallbackSupported;
};

class HardwareCapabilitiesExport final : public HardwareCapabilities
{
public:
    explicit HardwareCapabilitiesExport(DeviceVersion deviceVersionIn);
    virtual ~HardwareCapabilitiesExport() = default;

    HardwareCapabilitiesExport(const HardwareCapabilitiesExport&) = delete;
    HardwareCapabilitiesExport(HardwareCapabilitiesExport&&) = delete;
    HardwareCapabilitiesExport& operator=(const HardwareCapabilitiesExport&) = delete;
    HardwareCapabilitiesExport& operator=(HardwareCapabilitiesExport&&) = delete;
};

}
