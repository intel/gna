/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef WIN32

#include "LinuxDriverInterface.h"

#include "GnaException.h"
#include "HardwareRequest.h"
#include "Memory.h"
#include "Request.h"

#include "gna-h-wrapper.h"

#include "gna-api.h"
#include "gna-api-status.h"
#include "profiler.h"

#include "gna2-common-impl.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace GNA;

bool LinuxDriverInterface::OpenDevice(uint32_t deviceIndex)
{
    union gna_parameter params[] =
    {
        { GNA_PARAM_DEVICE_TYPE },
        { GNA_PARAM_INPUT_BUFFER_S },
        { GNA_PARAM_RECOVERY_TIMEOUT },
    };
    constexpr size_t paramsNum = sizeof(params)/sizeof(params[0]);

    const auto found = discoverDevice(deviceIndex, params, paramsNum);
    if (found == -1)
    {
        return false;
    }
    gnaFileDescriptor = found;

    try
    {
        driverCapabilities.deviceVersion = Gna2DeviceVersionFromInt(params[0].out.value);
        driverCapabilities.recoveryTimeout = static_cast<uint32_t>(params[2].out.value);
        driverCapabilities.hwInBuffSize = static_cast<uint32_t>(params[1].out.value);
    }
    catch(std::out_of_range&)
    {
        return false;
    }

    return true;
}

LinuxDriverInterface::~LinuxDriverInterface()
{
    if (gnaFileDescriptor != -1)
    {
        close(gnaFileDescriptor);
    }
}

uint64_t LinuxDriverInterface::MemoryMap(void *memory, uint32_t memorySize)
{
    union gna_memory_map memory_map;

    memory_map.in.address = reinterpret_cast<uint64_t>(memory);
    memory_map.in.size = memorySize;

    if(ioctl(gnaFileDescriptor, GNA_MAP_MEMORY, &memory_map) != 0)
    {
        throw GnaException {Gna2StatusDeviceOutgoingCommunicationError};
    }

    return memory_map.out.memory_id;
}

void LinuxDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    if(ioctl(gnaFileDescriptor, GNA_UNMAP_MEMORY, memoryId) != 0)
    {
        throw GnaException {Gna2StatusDeviceOutgoingCommunicationError};
    }
}

RequestResult LinuxDriverInterface::Submit(HardwareRequest& hardwareRequest,
                                        RequestProfiler * const profiler) const
{
    RequestResult result = { };
    int ret;

    createRequestDescriptor(hardwareRequest);

    gna_compute computeArgs;
    computeArgs.in.config = *reinterpret_cast<struct gna_compute_cfg *>(hardwareRequest.CalculationData.get());
    auto computeConfig = &computeArgs.in.config;

    computeConfig->gna_mode = hardwareRequest.Mode == xNN ? GNA_MODE_XNN : GNA_MODE_GMM;
    computeConfig->layer_count = hardwareRequest.LayerCount;

    if(xNN == hardwareRequest.Mode)
    {
        computeConfig->layer_base = hardwareRequest.LayerBase;
    }
    else if(GMM == hardwareRequest.Mode)
    {
        computeConfig->layer_base = hardwareRequest.GmmOffset;
        computeConfig->active_list_on = hardwareRequest.GmmModeActiveListOn ? 1 : 0;
    }
    else
    {
        throw GnaException { Gna2StatusXnnErrorLyrCfg };
    }

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestReady);

    ret = ioctl(gnaFileDescriptor, GNA_COMPUTE, &computeArgs);
    if (ret == -1)
    {
        throw GnaException { Gna2StatusDeviceOutgoingCommunicationError };
    }

    gna_wait wait_data = {};
    wait_data.in.request_id = computeArgs.out.request_id;
    wait_data.in.timeout = (driverCapabilities.recoveryTimeout + 1) * 1000;

    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestSent);
    ret = ioctl(gnaFileDescriptor, GNA_WAIT, &wait_data);
    profiler->Measure(Gna2InstrumentationPointLibDeviceRequestCompleted);
    if(ret == 0)
    {
        result.status = ((wait_data.out.hw_status & GNA_STS_SATURATE) != 0)
            ? Gna2StatusWarningArithmeticSaturation
            : Gna2StatusSuccess;

        result.driverPerf.Preprocessing = wait_data.out.drv_perf.pre_processing;
        result.driverPerf.Processing = wait_data.out.drv_perf.processing;
        result.driverPerf.DeviceRequestCompleted = wait_data.out.drv_perf.hw_completed;
        result.driverPerf.Completion = wait_data.out.drv_perf.completion;

        const auto profilerConfiguration = hardwareRequest.GetProfilerConfiguration();
        if (profilerConfiguration)
        {
            convertDriverPerfResult(profilerConfiguration->GetUnit(), result.driverPerf);
        }

        result.hardwarePerf.total = wait_data.out.hw_perf.total;
        result.hardwarePerf.stall = wait_data.out.hw_perf.stall;
    }
    else
    {
        switch(errno)
        {
        case EIO:
            result.status = parseHwStatus(static_cast<uint32_t>(wait_data.out.hw_status));
            break;
        case EBUSY:
            result.status = Gna2StatusWarningDeviceBusy;
            break;
        case ETIME:
            result.status = Gna2StatusDeviceCriticalFailure;
            break;
        default:
            result.status = Gna2StatusDeviceIngoingCommunicationError;
            break;
        }
    }

    return result;
}

void LinuxDriverInterface::createRequestDescriptor(HardwareRequest& hardwareRequest) const
{
    auto& scoreConfigSize = hardwareRequest.CalculationSize;
    scoreConfigSize = sizeof(struct gna_compute_cfg);

    for (const auto &buffer : hardwareRequest.DriverMemoryObjects)
    {
        scoreConfigSize += sizeof(struct gna_buffer) +
            buffer.Patches.size() * sizeof(struct gna_memory_patch);
    }

    scoreConfigSize = RoundUp(scoreConfigSize, sizeof(uint64_t));
    hardwareRequest.CalculationData.reset(new uint8_t[scoreConfigSize]);

    uint8_t *calculationData = static_cast<uint8_t *>(hardwareRequest.CalculationData.get());
    auto computeConfig = reinterpret_cast<struct gna_compute_cfg *>(
                        hardwareRequest.CalculationData.get());
    memset(computeConfig, 0, scoreConfigSize);
    computeConfig->hw_perf_encoding = hardwareRequest.HwPerfEncoding;

    computeConfig->buffers_ptr = reinterpret_cast<uintptr_t>(
                                calculationData + sizeof(struct gna_compute_cfg));
    computeConfig->buffer_count = hardwareRequest.DriverMemoryObjects.size();

    auto buffer = reinterpret_cast<struct gna_buffer *>(computeConfig->buffers_ptr);
    auto patch = reinterpret_cast<struct gna_memory_patch *>(computeConfig->buffers_ptr +
                    computeConfig->buffer_count * sizeof(struct gna_buffer));

    for (const auto &driverBuffer : hardwareRequest.DriverMemoryObjects)
    {
        buffer->memory_id = driverBuffer.Buffer.GetId();
        buffer->offset = 0;
        buffer->size = driverBuffer.Buffer.GetSize();
        buffer->patches_ptr = reinterpret_cast<uintptr_t>(patch);
        buffer->patch_count = driverBuffer.Patches.size();

        for (const auto &driverPatch : driverBuffer.Patches)
        {
            patch->offset = driverPatch.Offset;
            patch->size = driverPatch.Size;
            patch->value = driverPatch.Value;
            patch++;
        }

        buffer++;
    }

    hardwareRequest.SubmitReady = true;
}

Gna2Status LinuxDriverInterface::parseHwStatus(uint32_t hwStatus) const
{
    if ((hwStatus & GNA_STS_PCI_MMU_ERR) != 0)
    {
        return Gna2StatusDeviceMmuRequestError;
    }
    if ((hwStatus & GNA_STS_PCI_DMA_ERR) != 0)
    {
        return Gna2StatusDeviceDmaRequestError;
    }
    if ((hwStatus & GNA_STS_PCI_UNEXCOMPL_ERR) != 0)
    {
        return Gna2StatusDeviceUnexpectedCompletion;
    }
    if ((hwStatus & GNA_STS_VA_OOR) != 0)
    {
        return Gna2StatusDeviceVaOutOfRange;
    }
    if ((hwStatus & GNA_STS_PARAM_OOR) != 0)
    {
        return Gna2StatusDeviceParameterOutOfRange;
    }

    return Gna2StatusDeviceCriticalFailure;
}

int LinuxDriverInterface::discoverDevice(uint32_t deviceIndex, gna_parameter *params, size_t paramsNum)
{
    int fd = -1;
    uint32_t found = 0;
    for (uint8_t i = 0; i < MAX_GNA_DEVICES; i++)
    {
        char name[12];
        sprintf(name, "/dev/gna%hhu", i);
        fd = open(name, O_RDWR);
        if (-1 == fd)
        {
            continue;
        }

        bool paramsValid = true;
        for (size_t p = 0; p < paramsNum && paramsValid; p++)
        {
            paramsValid &= ioctl(fd, GNA_GET_PARAMETER, &params[p]) == 0;
        }
        if (paramsValid && found++ == deviceIndex)
        {
            return fd;
        }

        close(fd);
        fd = -1;
    }
    return -1;
}

 void LinuxDriverInterface::convertDriverPerfResult(
     const Gna2InstrumentationUnit targetUnit, DriverPerfResults & driverPerf)
{
    uint64_t divider = 1;

    switch (targetUnit)
    {
    case Gna2InstrumentationUnitMicroseconds:
        divider = 1000;
        break;
    case Gna2InstrumentationUnitMilliseconds:
        divider = 1000000;
        break;
    case Gna2InstrumentationUnitCycles:
        // Linux driver of GNA does not provide this data in cycles
        driverPerf.Completion = 0;
        driverPerf.DeviceRequestCompleted = 0;
        driverPerf.Preprocessing = 0;
        driverPerf.Processing = 0;
        return;
    }

    const auto cmpl = driverPerf.Completion;
    const auto devreq = driverPerf.DeviceRequestCompleted;
    const auto prepr = driverPerf.Preprocessing;
    const auto proc = driverPerf.Processing;
    const auto round = divider/2;

    driverPerf.Preprocessing = 0;
    driverPerf.Processing = (proc - prepr + round) / divider;
    driverPerf.DeviceRequestCompleted = (devreq - prepr + round) / divider;
    driverPerf.Completion = (cmpl - prepr + round) / divider;
}

#endif // not defined WIN32
