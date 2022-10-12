/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#ifndef WIN32

#include "LinuxDriverInterface.h"

#include "GnaException.h"
#include "HardwareRequest.h"
#include "Memory.h"
#include "Request.h"

#include "gna2-common-impl.h"
#include "gna2-memory-impl.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <linux/limits.h>
#include <memory>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <xf86drm.h>

using namespace GNA;

bool LinuxDriverInterface::OpenDevice(uint32_t deviceIndex)
{
    LinuxDriverInterface::ParamsMap params = {
        { GNA_PARAM_DEVICE_TYPE,      { { { GNA_PARAM_DEVICE_TYPE } }, false } },
        { GNA_PARAM_INPUT_BUFFER_S,   { { { GNA_PARAM_INPUT_BUFFER_S } }, false } },
        { GNA_PARAM_RECOVERY_TIMEOUT, { { { GNA_PARAM_RECOVERY_TIMEOUT } }, false } },
        // 'old' driver has no knowledge of GNA_PARAM_DDI_VERSION,
        // hence GNA_PARAM_DDI_VERSION_UNKNOWN = 0 is returned.
        { GNA_PARAM_DDI_VERSION,      { { { GNA_PARAM_DDI_VERSION } }, true } },
    };

    const auto devFd = discoverDevice(deviceIndex, params);
    if (devFd == -1)
        return false;

    gnaFileDescriptor = devFd;

    auto paramValue = [&params](ParamsMap::key_type id) {
                          return (params[id].first.out.value);
                      };

    driverCapabilities.deviceVersion = static_cast<DeviceVersion>(paramValue(GNA_PARAM_DEVICE_TYPE));
    driverCapabilities.recoveryTimeout = static_cast<uint32_t>(paramValue(GNA_PARAM_RECOVERY_TIMEOUT));
    driverCapabilities.hwInBuffSize = static_cast<uint32_t>(paramValue(GNA_PARAM_INPUT_BUFFER_S));
    driverCapabilities.isSoftwareFallbackSupported = paramValue(GNA_PARAM_DDI_VERSION) >= GNA_DDI_VERSION_3;

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
    UNREFERENCED_PARAMETER(memorySize);

    // already mapped at memory creation time. Here only id is returned.
    auto it = drmGemObjects.find(memory);

    if (it == drmGemObjects.end())
        throw GnaException { Gna2StatusIdentifierInvalid };

    return it->second.handle;
}

bool LinuxDriverInterface::MemoryUnmap(uint64_t memoryId)
{
    Expect::InRange(memoryId,
                    static_cast<uint64_t>(std::numeric_limits<decltype(gna_mem_id::handle)>::max()),
                    Gna2StatusIdentifierInvalid);
    gemFree(static_cast<decltype(gna_mem_id::handle)>(memoryId));

    return true;
}

RequestResult LinuxDriverInterface::Submit(HardwareRequest& hardwareRequest,
                                           RequestProfiler & profiler) const
{
    RequestResult result = { };
    int ret;

    createRequestDescriptor(hardwareRequest);

    if (!buffersOriginFromDeviceValid(hardwareRequest.DriverMemoryObjects))
        throw GnaException { Gna2StatusIdentifierInvalid };

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

    profiler.Measure(Gna2InstrumentationPointLibDeviceRequestReady);

    if (hardwareRequest.IsSwFallbackEnabled())
        computeArgs.in.config.flags |= static_cast<decltype(computeArgs.in.config.flags)>(GNA_FLAG_SCORE_QOS);

    ret = ioctl(gnaFileDescriptor, DRM_IOCTL_GNA_COMPUTE, &computeArgs);
    if (ret == -1)
    {
        switch (errno)
        {
            case EBUSY:
                if (hardwareRequest.IsSwFallbackEnabled())
                    throw GnaException { Gna2StatusDeviceQueueError };
                break;

            default:
                throw GnaException { Gna2StatusDeviceOutgoingCommunicationError };
        }
    }

    gna_wait wait_data = {};
    wait_data.in.request_id = computeArgs.out.request_id;
    wait_data.in.timeout = (driverCapabilities.recoveryTimeout + 1) * 1000;

    profiler.Measure(Gna2InstrumentationPointLibDeviceRequestSent);
    ret = ioctl(gnaFileDescriptor, DRM_IOCTL_GNA_WAIT, &wait_data);
    profiler.Measure(Gna2InstrumentationPointLibDeviceRequestCompleted);
    if(ret == 0)
    {
        result.status = ((wait_data.out.hw_status & GNA_STS_SATURATE) != 0)
                        ? Gna2StatusWarningArithmeticSaturation
                        : Gna2StatusSuccess;

        result.driverPerf.Preprocessing = wait_data.out.drv_perf.pre_processing;
        result.driverPerf.Processing = wait_data.out.drv_perf.processing;
        result.driverPerf.DeviceRequestCompleted = wait_data.out.drv_perf.hw_completed;
        result.driverPerf.Completion = wait_data.out.drv_perf.completion;

        result.hardwarePerf.total = wait_data.out.hw_perf.total;
        result.hardwarePerf.stall = wait_data.out.hw_perf.stall;

        const auto profilerConfiguration = hardwareRequest.GetProfilerConfiguration();
        if (profilerConfiguration)
        {
            convertPerfResultUnit(result.driverPerf, profilerConfiguration->GetUnit());
            DriverInterface::convertPerfResultUnit(result.hardwarePerf, profilerConfiguration->GetUnit());
        }
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
        buffer->handle = static_cast<uint32_t>(driverBuffer.Buffer.GetId());
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

int LinuxDriverInterface::discoverDevice(uint32_t deviceIndex, ParamsMap &out)
{
    uint32_t devIt = 0;

    for (uint8_t i = 0; i < MAX_GNA_DEVICES; i++)
    {
        int devFd = gnaDevOpen(i);
        if (devFd < 0)
            continue;

        bool paramsValid = true;
        auto params = out;
        for (auto &it : params)
        {
            auto &value = it.second;
            int error = ioctl(devFd, DRM_IOCTL_GNA_GET_PARAMETER, &value.first);
            if (error < 0 && errno == EINVAL && value.second /*ZERO_ON_EINVAL*/)
            {
                value.first.out.value = 0;
                error = 0;
            }

            paramsValid &= !error;
        }
        if (paramsValid && devIt++ == deviceIndex)
        {
            out = std::move(params);
            return devFd;
        }

        close(devFd);
    }

    return -1;
}

void LinuxDriverInterface::convertPerfResultUnit(DriverPerfResults & driverPerf,
    const Gna2InstrumentationUnit targetUnit)
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

int LinuxDriverInterface::gnaDevOpen(int devNo)
{
    char buf[PATH_MAX];
    drmVersionPtr version;
    int fd;

    static constexpr int DrmFirstRenderDev = DRM_NODE_RENDER * 64;
    sprintf(buf, DRM_RENDER_DEV_NAME, DRM_DIR_NAME, DrmFirstRenderDev + devNo);

    if ((fd = open(buf, O_RDWR | O_CLOEXEC, 0)) < 0)
        return -1;

    if ((version = drmGetVersion(fd)))
    {
        bool isGna = std::string("gna") == version->name;
        drmFreeVersion(version);

        if (!isGna)
        {
            close(fd);
            return -1;
        }
    }
    else
    {
        close(fd);
        return -1;
    }

    return fd;
}

void* LinuxDriverInterface::gemAlloc(uint32_t &size)
{
    gna_gem_new createMemArgs { { size } };

    int ret = ioctl(gnaFileDescriptor, DRM_IOCTL_GNA_GEM_NEW, &createMemArgs);

    if (ret != 0)
        return nullptr;

    void* buffer =
            mmap(0, createMemArgs.out.size_granted, PROT_READ|PROT_WRITE, MAP_SHARED,
                 gnaFileDescriptor, createMemArgs.out.vma_fake_offset);

    if (buffer == MAP_FAILED)
    {
        gna_gem_free freeMemArgs { createMemArgs.out.handle };

        ioctl(gnaFileDescriptor, DRM_IOCTL_GNA_GEM_FREE, &freeMemArgs);
        // we don't care about IOCTL's return code, can't do anything better though.

        return nullptr;
    }

    drmGemObjects[buffer] = createMemArgs.out;
    size = static_cast<uint32_t>(createMemArgs.out.size_granted);

    return buffer;
}

Memory LinuxDriverInterface::MemoryCreate(uint32_t size, uint32_t ldSize)
{
    Expect::InRange(size, 1u, Memory::GNA_MAX_MEMORY_FOR_SINGLE_ALLOC, Gna2StatusMemorySizeInvalid);
    auto gemObj = gemAlloc(size);
    Expect::NotNull(gemObj, Gna2StatusResourceAllocationError);
    return Memory(gemObj, size, ldSize);
}

void LinuxDriverInterface::gemFree(__u32 handle)
{
    auto it = std::find_if(drmGemObjects.begin(), drmGemObjects.end(), [handle] (auto &memObj) { return memObj.second.handle == handle; });

    if (it == drmGemObjects.end())
        throw GnaException { Gna2StatusMemoryBufferInvalid };

    if (munmap(it->first, it->second.size_granted) != 0)
        throw GnaException { Gna2StatusDeviceOutgoingCommunicationError };

    gna_gem_free freeMemArgs { it->second.handle };

    if (ioctl(gnaFileDescriptor, DRM_IOCTL_GNA_GEM_FREE, &freeMemArgs) != 0)
    {
        throw GnaException { Gna2StatusDeviceOutgoingCommunicationError };
    }

    drmGemObjects.erase(it);
}

bool LinuxDriverInterface::buffersOriginFromDeviceValid(std::vector<DriverBuffer> &driverMemoryObjects) const
{
    return std::all_of(driverMemoryObjects.begin(), driverMemoryObjects.end(),
                       [=](auto &driverBuffer)
                       {
                           return drmGemObjects.find(driverBuffer.Buffer.Get()) != drmGemObjects.end();
                       });
}

#endif // not defined WIN32
