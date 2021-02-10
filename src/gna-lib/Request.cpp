/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "CompiledModel.h"
#include "Request.h"
#include "RequestConfiguration.h"

#include <algorithm>
#include <cstring>
#include <memory>

struct KernelBuffers;

using namespace GNA;

Request::Request(RequestConfiguration& config, std::unique_ptr<RequestProfiler> profiler) :
    Configuration(config),
    Profiler{std::move(profiler)}
{
    auto callback = [&](KernelBuffers *buffers, RequestProfiler *profilerPtr)
    {
        return Configuration.Model.Score(Configuration, profilerPtr, buffers);
    };
    scoreTask = std::packaged_task<Gna2Status(KernelBuffers *buffers, RequestProfiler *profiler)>(callback);
    future = scoreTask.get_future();
}

Gna2Status Request::WaitFor(uint64_t milliseconds)
{
    auto const future_status = future.wait_for(std::chrono::milliseconds(milliseconds));
    switch (future_status)
    {
    case std::future_status::ready:
    {
        auto const score_status = future.get();
        Profiler->Measure(Gna2InstrumentationPointLibReceived);
        Profiler->SaveResults(Configuration.GetProfilerConfiguration());
        return score_status;
    }
    default:
        return Gna2StatusWarningDeviceBusy;
    }
}

RequestProfiler::RequestProfiler(bool initialize)
{
    if (initialize)
    {
        Points.resize(ProfilerConfiguration::GetMaxNumberOfInstrumentationPoints(), 0);
    }
}

void RequestProfiler::AddResults(Gna2InstrumentationPoint point, uint64_t result)
{
    Points.at(point) += result;
}

void MillisecondProfiler::Measure(Gna2InstrumentationPoint pointType)
{
    Points.at(pointType) = static_cast<uint64_t>(std::chrono::duration_cast<chronoMs>(chronoClock::now().time_since_epoch()).count());
}

void MicrosecondProfiler::Measure(Gna2InstrumentationPoint pointType)
{
    Points.at(pointType) = static_cast<uint64_t>(std::chrono::duration_cast<chronoUs>(chronoClock::now().time_since_epoch()).count());
}

void CycleProfiler::Measure(Gna2InstrumentationPoint pointType)
{
    getTsc(&Points.at(pointType));
}

void RequestProfiler::SaveResults(ProfilerConfiguration* config)
{
    uint32_t i = 0;
    for (const auto& selectedPoint: config->Points)
    {
        config->SetResult(i++, Points.at(selectedPoint));
    }
}

std::unique_ptr<RequestProfiler> RequestProfiler::Create(ProfilerConfiguration* config)
{
    if (nullptr == config)
    {
        return std::make_unique<DisabledProfiler>();
    }

    switch (config->GetUnit())
    {
    case Gna2InstrumentationUnitMicroseconds:
        return std::make_unique<MicrosecondProfiler>();
    case Gna2InstrumentationUnitMilliseconds:
        return std::make_unique<MillisecondProfiler>();
    case Gna2InstrumentationUnitCycles:
        return std::make_unique<CycleProfiler>();
    default:
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

uint64_t RequestProfiler::ConvertElapsedTime(uint64_t frequency, uint64_t multiplier,
    uint64_t start, uint64_t stop)
{
    auto const elapsedCycles = stop - start;
    auto const round = frequency / 2;
    auto elapsedMicroseconds = (elapsedCycles * multiplier + round) / frequency;
    return elapsedMicroseconds;
}

void DisabledProfiler::Measure(Gna2InstrumentationPoint point)
{
    UNREFERENCED_PARAMETER(point);
}
void DisabledProfiler::AddResults(Gna2InstrumentationPoint point, uint64_t result)
{
    UNREFERENCED_PARAMETER(point);
    UNREFERENCED_PARAMETER(result);
}
void DisabledProfiler::SaveResults(ProfilerConfiguration* config)
{
    UNREFERENCED_PARAMETER(config);
}
