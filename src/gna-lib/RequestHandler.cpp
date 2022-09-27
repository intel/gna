/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/


#include "Expect.h"
#include "GnaException.h"
#include "Request.h"
#include "RequestHandler.h"
#include "RequestConfiguration.h"

#include <chrono>
#include <future>
#include <utility>
#include "Logger.h"

using namespace GNA;

RequestHandler::~RequestHandler()
{
    try
    {
        clearRequestMap();
    }
    catch (...)
    {
        Log->Error("RequestHandler destructor failed.\n");
    }
}

uint32_t RequestHandler::GetNumberOfThreads() const
{
    return threadPool.GetNumberOfThreads();
}

void RequestHandler::ChangeNumberOfThreads(uint32_t threadCount)
{
    threadPool.SetNumberOfThreads(threadCount);
}

void RequestHandler::Enqueue(
    uint32_t *requestId,
    std::unique_ptr<Request> request)
{
    Expect::NotNull(requestId);
    auto r = request.get();
    {
        std::lock_guard<std::mutex> lockGuard(lock);

        if (requests.size() >= QueueLengthMax)
        {
            throw GnaException(Gna2StatusDeviceQueueError);
        }

        *requestId = assignRequestId();
        r->Id = *requestId;
        addRequest(std::move(request));
    }
    r->Profiler->Measure(Gna2InstrumentationPointLibSubmission);

    threadPool.Enqueue(r);
}

void RequestHandler::addRequest(std::unique_ptr<Request> request)
{
    auto insert = requests.emplace(request->Id, move(request));
    if (!insert.second)
    {
        throw GnaException(Gna2StatusResourceAllocationError);
    }
}

Gna2Status RequestHandler::WaitFor(const uint32_t requestId, const uint32_t milliseconds)
{
    auto request = extractRequestLocked(requestId);

    auto const status = request->WaitFor(milliseconds);
    if (Gna2StatusWarningDeviceBusy == status)
    {
        addRequestLocked(std::move(request));
    }
    return status;
}

void RequestHandler::StopRequests()
{
    threadPool.StopAndJoin();
}

bool RequestHandler::HasRequest(uint32_t requestId) const
{
    return requests.count(requestId) > 0;
}

std::unique_ptr<Request> RequestHandler::extractRequestLocked(const uint32_t requestId)
{
    std::lock_guard<std::mutex> lockGuard(lock);
    auto found = requests.find(requestId);
    if(found == requests.end())
    {
        throw GnaException(Gna2StatusIdentifierInvalid);
    }
    auto extracted = std::move(found->second);
    requests.erase(found);
    return extracted;
}

void RequestHandler::addRequestLocked(std::unique_ptr<Request> request)
{
    std::lock_guard<std::mutex> lockGuard(lock);
    addRequest(std::move(request));
}

uint32_t RequestHandler::assignRequestId()
{
    static uint32_t id;
    return id++;
}

void RequestHandler::clearRequestMap()
{
    std::lock_guard<std::mutex> lockGuard(lock);
    requests.clear();
}
