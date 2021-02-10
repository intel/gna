/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifdef WIN32

#include "WindowsHwModuleInterface.hpp"

#include "Expect.h"
#include "Logger.h"

using namespace GNA;

WindowsHwModuleInterface::WindowsHwModuleInterface(char const* moduleName)
{
    fullName = moduleName;
    fullName.append(".dll");
    hwModule = LoadLibrary(fullName.c_str());
    if (nullptr != hwModule)
    {
        ImportAllSymbols();
    }
    else
    {
        Log->Warning("HwModule (%s) library not found.\n", fullName.c_str());
    }
}

WindowsHwModuleInterface::~WindowsHwModuleInterface()
{
    if (nullptr != hwModule)
    {
        auto const status = FreeLibrary(hwModule);
        if (!status)
        {
            Log->Error("FreeLibrary failed!\n");
        }
    }
}

void* WindowsHwModuleInterface::getSymbolAddress(const std::string& symbolName)
{
    return reinterpret_cast<void*>(GetProcAddress(hwModule, symbolName.c_str()));
}

#endif
