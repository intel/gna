/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "GnaException.h"
#include "ThreadPool.h"


#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace GNA
{
class Request;

class RequestHandler
{
public:
    explicit RequestHandler() = default;

    ~RequestHandler();

    uint32_t GetNumberOfThreads() const;

    void ChangeNumberOfThreads(uint32_t threadCount);

    void Enqueue(
        uint32_t *requestId,
        std::unique_ptr<Request> request);

    Gna2Status WaitFor(const uint32_t requestId, const uint32_t milliseconds);

    void StopRequests();

    bool HasRequest(uint32_t requestId) const;

private:

    void clearRequestMap();

    std::unique_ptr<Request> extractRequestLocked(uint32_t requestId);
    void addRequestLocked(std::unique_ptr<Request> request);
    void addRequest(std::unique_ptr<Request> request);

    static uint32_t assignRequestId();

    std::unordered_map<uint32_t, std::unique_ptr<Request>> requests;
    std::mutex lock;
    ThreadPool threadPool;

    /** Maximum number of requests that can be enqueued before retrieval */
    static constexpr auto QueueLengthMax = 64u;
};

}
