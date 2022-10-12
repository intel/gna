/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#ifndef WIN32

#include "DriverInterface.h"
#include "gna-h-wrapper.h"

#include <cstdint>

union gna_parameter;

namespace GNA
{
class HardwareRequest;
class RequestProfiler;

class LinuxDriverInterface : public DriverInterface
{
public:
    LinuxDriverInterface() = default;
    LinuxDriverInterface(const LinuxDriverInterface &) = delete;
    LinuxDriverInterface& operator=(const LinuxDriverInterface&) = delete;

    bool OpenDevice(uint32_t deviceIndex) override;

    Memory MemoryCreate(uint32_t size, uint32_t ldSize = Memory::GNA_BUFFER_ALIGNMENT) override;

    uint64_t MemoryMap(void *memory, uint32_t memorySize) override;
    bool MemoryUnmap(uint64_t memoryId) override;

    RequestResult Submit(HardwareRequest& hardwareRequest,
                                RequestProfiler & profiler) const override;

    ~LinuxDriverInterface() override;

private:
    using ParamsMap = std::map<gna_param_id, std::pair<union gna_parameter, bool /*ZERO_ON_EINVAL*/>>;
    using DrmGemObjects = std::map<void*/*buffer*/, gna_mem_id>;

    // open /dev/dri/cardDEVNO device as GNA one
    // return fd file descriptor on success or -1 on error
    int gnaDevOpen(int devNo);
    static void convertPerfResultUnit(DriverPerfResults & driverPerf,
                                      const Gna2InstrumentationUnit targetUnit);
    void createRequestDescriptor(HardwareRequest& hardwareRequest) const override;
    Gna2Status parseHwStatus(uint32_t hwStatus) const override;
    bool buffersOriginFromDeviceValid(std::vector<DriverBuffer> &driverMemoryObjects) const;
    int discoverDevice(uint32_t deviceIndex, ParamsMap &out);
    void* gemAlloc(uint32_t &size);
    void gemFree(__u32 handle);

    DrmGemObjects drmGemObjects;
    int gnaFileDescriptor = -1;
};

}

#endif // !WIN32
