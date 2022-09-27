/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "gna2-common-impl.h"

struct GNA3_AdaptHW;
struct GNA3_LyrDesc;

namespace GNA
{
struct ConvolutionFunction2D;
struct DataMode;
class PoolingFunction2D;

struct HwUarchParams
{
    bool Valid;
    uint16_t KWG;
    uint16_t KWGIter;
    uint8_t uT;
    uint8_t KMemBase;
    uint8_t CMemBase;
    uint8_t PMemBase;

    HwUarchParams() = default;
    explicit HwUarchParams(struct GNA3_AdaptHW const& source);
};

class HwModuleInterface
{
public:
    HwModuleInterface(DeviceVersion deviceVersion);
    ~HwModuleInterface() = default;

    HwModuleInterface(const HwModuleInterface&) = delete;
    HwModuleInterface& operator=(const HwModuleInterface&) = delete;

    HwUarchParams GetCnnParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode, bool is1D) const;

    bool IsModuleLoaded() const;
protected:
    HwUarchParams Get1DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode) const;
    HwUarchParams Get2DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode) const;
    static int32_t GetPoolingMode(PoolingFunction2D const* poolingIn);

    bool SetConfig(DeviceVersion deviceVersion);

    bool isModuleLoaded = false;
};
}
