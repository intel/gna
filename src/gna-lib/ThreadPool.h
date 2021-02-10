/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "KernelArguments.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <deque>
#include <thread>
#include <vector>

namespace GNA
{
class Request;

class ThreadPool {
public:
    explicit ThreadPool(uint32_t threadCount);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    uint32_t GetNumberOfThreads() const;

    void SetNumberOfThreads(uint32_t threadCount);

    void Enqueue(Request *request);
    void StopAndJoin();

private:
    void employWorkers();

    // NOTE: order is important, buffers have to be destroyed last
    std::vector<KernelBuffers> buffers;
    std::mutex tpMutex;
    std::deque<Request*> tasks;
    bool stopped = false;
    std::condition_variable condition;
    std::vector<std::thread> workers;
    uint32_t numberOfThreads;
};

}
