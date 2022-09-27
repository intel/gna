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

    virtual ~LinuxDriverInterface() override;

    virtual bool OpenDevice(uint32_t deviceIndex) override;

    virtual uint64_t MemoryMap(void *memory, uint32_t memorySize) override;

    virtual void MemoryUnmap(uint64_t memoryId) override;

    virtual RequestResult Submit(HardwareRequest& hardwareRequest,
                                RequestProfiler & profiler) const override;

    LinuxDriverInterface(const LinuxDriverInterface &) = delete;
    LinuxDriverInterface& operator=(const LinuxDriverInterface&) = delete;

private:
    using ParamsMap = std::map<gna_param_id, std::pair<union gna_parameter, bool /*ZERO_ON_EINVAL*/>>;

    static void convertPerfResultUnit(DriverPerfResults & driverPerf,
        const Gna2InstrumentationUnit targetUnit);

    void createRequestDescriptor(HardwareRequest& hardwareRequest) const override;

    Gna2Status parseHwStatus(uint32_t hwStatus) const override;

    int discoverDevice(uint32_t deviceIndex, ParamsMap &out);

    int gnaFileDescriptor = -1;
};

}
#endif // !WIN32
