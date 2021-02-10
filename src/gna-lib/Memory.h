/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "Macros.h"

#include <cstdint>

namespace GNA
{
class DriverInterface;

class Memory : public BaseAddress
{
public:
    Memory() = default;

    // just makes object from arguments
    Memory(void * bufferIn, uint32_t userSize, uint32_t alignment = GNA_BUFFER_ALIGNMENT);

    // allocates and zeros memory
    Memory(const uint32_t userSize, uint32_t alignment = GNA_BUFFER_ALIGNMENT);

    virtual ~Memory();

    void Map(DriverInterface& ddi);
    void Unmap(DriverInterface& ddi);

    uint64_t GetId() const;

    uint32_t GetSize() const
    {
        return size;
    }

    template<class T = void> T * GetBuffer() const
    {
        return Get<T>();
    }

    static const uint32_t GNA_BUFFER_ALIGNMENT = 64;
    static constexpr uint32_t GNA_MAX_MEMORY_FOR_SINGLE_ALLOC = 1 << 28;

protected:
    uint64_t id = 0;

    uint32_t size = 0;

    bool mapped = false;

    bool deallocate = true;
};

}
