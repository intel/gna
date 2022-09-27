/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#ifdef WIN32

#include "DriverInterface.h"

#include "Request.h"
#include "Expect.h"

#include <map>
#include <memory>
#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS

namespace GNA
{

class WinHandle
{
public:
    WinHandle() :
        deviceHandle(INVALID_HANDLE_VALUE)
    {};

    explicit WinHandle(HANDLE const handle) :
        deviceHandle(handle)
    {};

    ~WinHandle()
    {
        if (INVALID_HANDLE_VALUE != deviceHandle)
        {
            CloseHandle(deviceHandle);
            deviceHandle = INVALID_HANDLE_VALUE;
        }
    }

    WinHandle(const WinHandle &) = delete;
    WinHandle& operator=(const WinHandle&) = delete;

    void Set(HANDLE const handle)
    {
        Expect::Equal(INVALID_HANDLE_VALUE, deviceHandle, Gna2StatusIdentifierInvalid);
        deviceHandle = handle;
    }

    operator HANDLE() const
    {
        return deviceHandle;
    }

private:
    HANDLE deviceHandle;
};

class OverlappedWithEvent
{
public:
    OverlappedWithEvent();
    OverlappedWithEvent(const OverlappedWithEvent&) = delete;
    OverlappedWithEvent(OverlappedWithEvent&&) = delete;
    OverlappedWithEvent& operator = (const OverlappedWithEvent&) = delete;
    OverlappedWithEvent& operator = (OverlappedWithEvent&&) = delete;
    ~OverlappedWithEvent();
    operator OVERLAPPED*()
    {
        return &overlapped;
    }
private:
    OVERLAPPED overlapped = {};
};

class WindowsDriverInterface : public DriverInterface
{
    static const int WAIT_FOR_MAP_ITERATIONS;
    static const int WAIT_FOR_MAP_MILLISECONDS;
    static const uint64_t FORBIDDEN_MEMORY_ID;
public:
    WindowsDriverInterface();

    virtual ~WindowsDriverInterface();

    virtual bool OpenDevice(uint32_t deviceIndex) override;

    virtual uint64_t MemoryMap(void *memory, uint32_t memorySize) override;

    virtual bool MemoryUnmap(uint64_t memoryId) override;

    virtual RequestResult Submit(
        HardwareRequest& hardwareRequest, RequestProfiler& profiler) const override;

protected:
    void createRequestDescriptor(HardwareRequest& hardwareRequest) const;

    Gna2Status parseHwStatus(uint32_t hwStatus) const override;

    void QoSRequest(size_t size, OverlappedWithEvent & ioHandle, void * input) const;

    template<Gna2Status status = Gna2StatusDeviceOutgoingCommunicationError>
    void InferenceRequest(size_t size, OverlappedWithEvent & ioHandle, void * input, char const message[] = nullptr) const
    {
        auto const ioResult = WriteFile(deviceHandle,
            input, static_cast<DWORD>(size),
            nullptr, ioHandle);
        checkStatus<status>(ioResult, message);
    }

private:
    WindowsDriverInterface(const WindowsDriverInterface &) = delete;
    WindowsDriverInterface& operator=(const WindowsDriverInterface&) = delete;

    inline static std::string lastErrorToString(DWORD error);

    void wait(LPOVERLAPPED ioctl) const;

    void verify(LPOVERLAPPED ioctl) const;

    template<Gna2Status status = Gna2StatusDeviceOutgoingCommunicationError>
    void checkStatus(BOOL ioResult, char const message[] = nullptr) const
    {
        throwOnFailedPredicate(
            [](BOOL ioResult, DWORD error)
        {return (ioResult == 0 && ERROR_IO_PENDING != error); },
            0,
            status,
            message);
    }

    template <typename Predicate>
    void getOverlappedResult(Predicate predicate,
        LPOVERLAPPED ioctl,
        DWORD timeout,
        Gna2Status status,
        char const * message) const
    {
        auto bytesRead = DWORD{ 0 };
        auto const ioResult = GetOverlappedResultEx(deviceHandle,
            ioctl,
            &bytesRead,
            timeout,
            false);
        throwOnFailedPredicate(predicate,
            ioResult,
            status,
            message);
    }

    template <typename Predicate>
    void throwOnFailedPredicate(Predicate predicate,
        BOOL ioResult,
        Gna2Status status,
        char const * message) const
    {
        auto const error = GetLastError();
        if (predicate(ioResult, error))
        {
            if (message)
            {
                Log->Error(message);
#if DEBUG == 1
                auto const errorDescription = lastErrorToString(error);
                Log->Error("%s\n", errorDescription.c_str());
#endif
            }
            throw GnaException(status);
        }
        // io completed successfully
    }

    void getDeviceCapabilities();

    void getDeviceCapabilityRequest(UINT64 param, UINT64 & value, bool newDDI);

    static uint64_t getPerfCounterFrequency();

    static std::string discoverDevice(uint32_t deviceIndex);

    static const std::map<GnaIoctlCommand, DWORD> ioctlCommandsMap;

    WinHandle deviceHandle;
    WinHandle deviceEvent;
    OVERLAPPED overlapped;

    UINT32 recoveryTimeout;

    std::map<uint64_t, std::unique_ptr<OverlappedWithEvent>> memoryMapRequests;
};

}

#endif // WIN32
