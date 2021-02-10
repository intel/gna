/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "RequestBuilder.h"

#include "RequestConfiguration.h"
#include "GnaException.h"
#include "Request.h"

#include "gna-api-status.h"
#include "profiler.h"

#include <algorithm>
#include <stdexcept>

namespace GNA
{
class CompiledModel;
struct ActiveList;
}

using namespace GNA;

uint32_t RequestBuilder::assignConfigId()
{
    static uint32_t configIdSequence = 0;
    return configIdSequence++;
}

void RequestBuilder::CreateConfiguration(CompiledModel& model, uint32_t *configId, DeviceVersion consistentDevice)
{
    Expect::NotNull(configId);
    *configId = assignConfigId();
    configurations.emplace(*configId, std::make_unique<RequestConfiguration>(model, *configId, consistentDevice));
}

void RequestBuilder::ReleaseConfiguration(uint32_t configId)
{
    configurations.erase(configId);
}

void RequestBuilder::AttachBuffer(uint32_t configId, uint32_t operandIndex, uint32_t layerIndex,
    void * address) const
{
    auto& configuration = GetConfiguration(configId);
    configuration.AddBuffer(operandIndex, layerIndex, address);
}

void RequestBuilder::AttachActiveList(uint32_t configId, uint32_t layerIndex,
    const ActiveList& activeList) const
{
    auto& configuration = GetConfiguration(configId);
    configuration.AddActiveList(layerIndex, activeList);
}

RequestConfiguration& RequestBuilder::GetConfiguration(uint32_t configId) const
{
    try
    {
        auto& config = configurations.at(configId);
        return *config;
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

std::unique_ptr<Request> RequestBuilder::CreateRequest(uint32_t configId)
{
    auto& configuration = GetConfiguration(configId);
    auto profiler = RequestProfiler::Create(configuration.GetProfilerConfiguration());
    profiler->Measure(Gna2InstrumentationPointLibPreprocessing);

    return std::make_unique<Request>(configuration, std::move(profiler));
}

uint32_t RequestBuilder::AssignProfilerConfigId()
{
    return profilerConfigIdSequence++;
}

uint32_t RequestBuilder::CreateProfilerConfiguration(
    std::vector<Gna2InstrumentationPoint>&& selectedInstrumentationPoints,
    uint64_t* results)
{
    auto const profilerConfigId = AssignProfilerConfigId();
    profilerConfigurations.emplace(profilerConfigId,
        std::make_unique<ProfilerConfiguration>(profilerConfigId, std::move(selectedInstrumentationPoints), results));
    return profilerConfigId;
}

ProfilerConfiguration& RequestBuilder::GetProfilerConfiguration(uint32_t configId) const
{
    try
    {
        auto& config = profilerConfigurations.at(configId);
        return *config.get();
    }
    catch (const std::out_of_range&)
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
}

void RequestBuilder::ReleaseProfilerConfiguration(uint32_t configId)
{
    profilerConfigurations.erase(configId);
}

bool RequestBuilder::HasConfiguration(uint32_t configId) const
{
    return configurations.count(configId) > 0;
}
