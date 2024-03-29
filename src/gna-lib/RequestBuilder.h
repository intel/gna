/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "ProfilerConfiguration.h"
#include "RequestConfiguration.h"

#include "gna2-instrumentation-api.h"

#include <memory>
#include <cstdint>
#include <unordered_map>

namespace GNA
{
class CompiledModel;
class Request;
struct ActiveList;

class RequestBuilder
{
public:
    RequestBuilder() = default;
    RequestBuilder(const RequestBuilder &) = delete;
    RequestBuilder& operator=(const RequestBuilder&) = delete;

    void CreateConfiguration(CompiledModel& model, uint32_t *configId, const HardwareCapabilities & hardwareCapabilities);
    void ReleaseConfiguration(uint32_t configId);

    void AttachBuffer(uint32_t configId, uint32_t operandIndex, uint32_t layerIndex, void * address);

    void AttachActiveList(uint32_t configId, uint32_t layerIndex, const ActiveList& activeList);

    RequestConfiguration& GetConfiguration(uint32_t configId);

    std::unique_ptr<Request> CreateRequest(uint32_t configId);

    bool HasConfiguration(uint32_t configId) const;

private:
    std::unordered_map<uint32_t, std::unique_ptr<RequestConfiguration>> configurations;
    static uint32_t assignConfigId();
};

}
