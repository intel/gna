/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "HwModuleInterface.hpp"

#include "ConvolutionalFunctions.h"
#include "ConvolutionalFunctions2D.h"
#include "DataMode.h"
#include "LinuxHwModuleInterface.hpp"
#include "Logger.h"
#include "PoolingFunctions2D.h"
#include "WindowsHwModuleInterface.hpp"

#include <cstdint>
#include <memory>

#undef GNA_HW_MODULE_CLASS
#if defined(_WIN32)
#   define GNA_HW_MODULE_CLASS WindowsHwModuleInterface
#else // GNU/Linux / Android / ChromeOS
#   define GNA_HW_MODULE_CLASS LinuxHwModuleInterface
#endif

using namespace GNA;

HwUarchParams::HwUarchParams(struct GNA3_AdaptHW const& source)
{
    UNREFERENCED_PARAMETER(source);
}

std::unique_ptr<HwModuleInterface const> HwModuleInterface::Create(char const* moduleName)
{
    Expect::NotNull(moduleName);
    Expect::False(std::string(moduleName).empty(), Gna2StatusAccelerationModeNotSupported);
    try
    {
        return std::make_unique<GNA_HW_MODULE_CLASS const>(moduleName);
    }
    catch (GnaException & e)
    {
        Log->Warning("HwModule library load failed.");
        throw e;
    }
}

HwUarchParams HwModuleInterface::GetCnnParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                              const DataMode& outputMode, bool is1D) const
{
    Expect::True(libraryLoadSuccess, Gna2StatusHardwareModuleNotFound);
    Expect::True(symbolImportSuccess, Gna2StatusHardwareModuleSymbolNotFound);
    Expect::NotNull(cnnIn);

    if (is1D)
    {
        return Get1DParams(cnnIn, poolingIn, outputMode);
    }
    return Get2DParams(cnnIn, poolingIn, outputMode);
}

int32_t HwModuleInterface::GetPoolingMode(PoolingFunction2D const* poolingIn)
{
    if (poolingIn == nullptr)
    {
        return static_cast<int32_t>(KernelPoolingModeNone);
    }
    return static_cast<int32_t>(poolingIn->Mode);
}

HwUarchParams HwModuleInterface::Get1DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                             const DataMode& outputMode) const
{
    UNREFERENCED_PARAMETER(cnnIn);
    UNREFERENCED_PARAMETER(poolingIn);
    UNREFERENCED_PARAMETER(outputMode);
    throw GnaException(Gna2StatusNotImplemented);
}

HwUarchParams HwModuleInterface::Get2DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                             const DataMode& outputMode) const
{
    UNREFERENCED_PARAMETER(cnnIn);
    UNREFERENCED_PARAMETER(poolingIn);
    UNREFERENCED_PARAMETER(outputMode);
    throw GnaException(Gna2StatusNotImplemented);
}

void HwModuleInterface::ImportAllSymbols()
{
    libraryLoadSuccess = true;
    symbolImportSuccess = true;
    Log->Message("HwModule library (%s) loaded successfully.\n", fullName.c_str());
    CreateLD = reinterpret_cast<CreateLDFunction>(GetSymbolAddress("GNA3_NewLD"));
    FillLD = reinterpret_cast<FillLDFunction>(GetSymbolAddress("GNA3_PopLD"));
    FreeLD = reinterpret_cast<FreeLDFunction>(GetSymbolAddress("GNA3_FreeLD"));
}

void* HwModuleInterface::GetSymbolAddress(const std::string& symbolName)
{
    const auto ptr = getSymbolAddress(symbolName);
    if(ptr == nullptr)
    {
        Log->Warning("HwModule library (%s), symbol (%s) not found.\n", fullName.c_str(), symbolName.c_str());
        symbolImportSuccess = false;
    }
    return ptr;
}
