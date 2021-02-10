/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "HardwareModel.h"

#include "Address.h"
#include "HardwareCapabilities.h"

#include <cstdint>
#include <memory>

struct Gna2ModelSueCreekHeader;

namespace GNA
{
    class LayerDescriptor;

    class HardwareModelSue1 : public HardwareModel
    {
    public:

        HardwareModelSue1(CompiledModel const & softwareModel, Gna2UserAllocator customAlloc);

        virtual ~HardwareModelSue1() = default;

        const LayerDescriptor& GetDescriptor(uint32_t layerIndex) const;

        uint32_t GetOutputOffset(uint32_t layerIndex) const;

        uint32_t GetInputOffset(uint32_t layerIndex) const;

        // this override does not add PAGE_SIZE alignment to calculations
        // since memory buffers are copied to one allocated memory buffer
        virtual uint32_t GetBufferOffset(const BaseAddress& address) const override;

        void * Export();

        void PopulateHeader(Gna2ModelSueCreekHeader& modelHeader) const;

    protected:
        virtual void prepareAllocationsAndModel() override;

    private:
        static HardwareCapabilities sueCapabilities;

        Gna2UserAllocator customAlloc = nullptr;

        void * exportMemory = nullptr;

        uint32_t totalModelSize = 0;

        std::unique_ptr<Memory> scratchPadMemory;

    };

}
