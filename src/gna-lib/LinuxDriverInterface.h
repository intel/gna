/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "DriverInterface.h"

#include "common.h"

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
                                RequestProfiler * const profiler) const override;

    LinuxDriverInterface(const LinuxDriverInterface &) = delete;
    LinuxDriverInterface& operator=(const LinuxDriverInterface&) = delete;

private:
    static void convertDriverPerfResult(Gna2InstrumentationUnit targetUnit, DriverPerfResults & driverPerf);

    void createRequestDescriptor(HardwareRequest& hardwareRequest) const override;

    Gna2Status parseHwStatus(uint32_t hwStatus) const override;

    int discoverDevice(uint32_t deviceIndex, gna_parameter *params, size_t paramsNum);

    int gnaFileDescriptor = -1;
};

}
