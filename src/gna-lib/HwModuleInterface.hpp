/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

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
    /**
     * Create HW Module for underlying OS.
     * 
     * @param moduleName Name of library without path and extension.
     */
    static std::unique_ptr<HwModuleInterface const> Create(char const* moduleName);

    HwModuleInterface(const HwModuleInterface&) = delete;
    HwModuleInterface& operator=(const HwModuleInterface&) = delete;
    virtual ~HwModuleInterface() = default;

    HwUarchParams GetCnnParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode, bool is1D) const;

protected:
    HwModuleInterface() = default;


    HwUarchParams Get1DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode) const;
    HwUarchParams Get2DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
        const DataMode& outputMode) const;
    static int32_t GetPoolingMode(PoolingFunction2D const* poolingIn);

    typedef struct GNA3_LyrDesc* (*CreateLDFunction)();
    typedef void(*FreeLDFunction)(struct GNA3_LyrDesc* LD);
    typedef bool(*FillLDFunction)(struct GNA3_LyrDesc* LD);

    CreateLDFunction CreateLD = nullptr;
    FreeLDFunction FreeLD = nullptr;
    FillLDFunction FillLD = nullptr;

    std::string fullName;

    void ImportAllSymbols();
    void* GetSymbolAddress(const std::string& symbolName);
    bool libraryLoadSuccess = false;
    bool symbolImportSuccess = false;
private:
    virtual void* getSymbolAddress(const std::string& symbolName) = 0;
};
}
