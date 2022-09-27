/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-common-api.h"
#include "gna2-instrumentation-api.h"

#include <future>
#include <memory>
#include <vector>

struct KernelBuffers;

namespace GNA
{
    class ProfilerConfiguration;
    class RequestConfiguration;
    class RequestProfiler;

/**
 * Library level request processing profiler
 */
class RequestProfiler
{
public:
    static const uint64_t MICROSECOND_MULTIPLIER = 1000000;
    static const uint64_t MILLISECOND_MULTIPLIER = 1000;
    static const uint64_t DEVICE_CLOCK_FREQUENCY = 400'000'000;

    static std::unique_ptr<RequestProfiler> Create(ProfilerConfiguration* config);

    RequestProfiler(bool initialize = true);
    virtual ~RequestProfiler() = default;

    virtual void AddResults(Gna2InstrumentationPoint point, uint64_t result);

    virtual void Measure(Gna2InstrumentationPoint point) = 0;

    virtual void SaveResults(ProfilerConfiguration* config);

    static uint64_t ConvertElapsedTime(uint64_t frequency, uint64_t multiplier,
        uint64_t start, uint64_t stop);
protected:

    std::vector<uint64_t> Points;
}; // Library level request processing profiler

class DisabledProfiler : public RequestProfiler
{
public:
    DisabledProfiler() : RequestProfiler(false)
    {

    }

    void Measure(Gna2InstrumentationPoint point) override;
    void AddResults(Gna2InstrumentationPoint point, uint64_t result) override;
    void SaveResults(ProfilerConfiguration* config) override;
};

class MicrosecondProfiler : public RequestProfiler
{
public:
    void Measure(Gna2InstrumentationPoint point) override;
};

class MillisecondProfiler : public RequestProfiler
{
public:
    void Measure(Gna2InstrumentationPoint point) override;
};

class CycleProfiler : public RequestProfiler
{
public:
    void Measure(Gna2InstrumentationPoint point) override;
};

/**
 * Calculation request for single scoring or propagate forward operation
 */
class Request
{
public:
    Request(RequestConfiguration& config, std::unique_ptr<RequestProfiler> profiler);
    ~Request() = default;
    Request() = delete;
    Request(const Request &) = delete;
    Request& operator=(const Request&) = delete;

    Gna2Status WaitFor(uint64_t milliseconds);

    void operator()(KernelBuffers *buffers)
    {
        scoreTask(buffers, *Profiler);
    }

    // External id (0-GNA_REQUEST_WAIT_ANY)
    uint32_t Id = 0;
    RequestConfiguration& Configuration;

    std::unique_ptr<RequestProfiler> Profiler;

private:
    std::packaged_task<Gna2Status(KernelBuffers *buffers, RequestProfiler &profiler)> scoreTask;

    std::future<Gna2Status> future;
};

}
