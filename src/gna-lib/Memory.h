/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "gna2-model-export-api.h"

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#endif
#include <cstdint>

namespace GNA
{

/** GNA main memory required alignment size */
constexpr auto MemoryBufferAlignment = uint32_t{ 0x1000 };

/** Allocator with alignment for HW data buffers */
#define _gna_malloc(a)    _mm_malloc(a, MemoryBufferAlignment)
/** Allocator with alignment for intrinsics */
#define _kernel_malloc(a) _mm_malloc(a, 0x40)
#define _gna_free(a)      _mm_free(a)

class DriverInterface;

class Memory : public BaseAddress
{
public:
    // just makes object from arguments
    Memory(void * bufferIn, uint32_t userSize, uint32_t alignment = GNA_BUFFER_ALIGNMENT);

    // allocates and zeros memory
    Memory(const uint32_t userSize, uint32_t alignment = GNA_BUFFER_ALIGNMENT);

    Memory(const Memory&) = delete;
    Memory(Memory&&) = default;
    Memory& operator=(const Memory&) = delete;
    Memory& operator=(Memory&&) = delete;

    virtual ~Memory();

    void Map(DriverInterface& ddi);
    // return 'true' if object has also been unallocated
    bool Unmap(DriverInterface& ddi);

    uint64_t GetId() const;

    uint32_t GetSize() const
    {
        return size;
    }

    template<class T = void> T * GetBuffer() const
    {
        return Get<T>();
    }

    void SetTag(uint32_t newTag);

    Gna2MemoryTag GetMemoryTag() const;

    static const uint32_t GNA_BUFFER_ALIGNMENT = 64;
    static constexpr uint32_t GNA_MAX_MEMORY_FOR_SINGLE_ALLOC = 1 << 28;

protected:
    uint64_t id = 0;

    uint32_t size = 0;

    uint32_t tag = 0;

    bool mapped = false;

    bool allocationOwner = true;
};

}
