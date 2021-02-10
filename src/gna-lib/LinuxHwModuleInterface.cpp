/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef WIN32

#include "LinuxHwModuleInterface.hpp"

#include "Expect.h"
#include "Logger.h"

#include <dlfcn.h>

using namespace GNA;

LinuxHwModuleInterface::LinuxHwModuleInterface(char const * moduleName)
{
    const auto prefixName = std::string("./") + moduleName;
    fullName = prefixName + ".so";
    hwModule = dlopen(fullName.c_str(), RTLD_NOW);
    if (nullptr == hwModule)
    {
        Log->Warning("HwModule release library (%s) not found, trying to load debug library.\n", fullName.c_str());
        fullName = prefixName + "d.so";
        hwModule = dlopen(fullName.c_str(), RTLD_NOW);
    }
    if (nullptr != hwModule)
    {
        ImportAllSymbols();
    }
    else
    {
        Log->Warning("HwModule (%s) library not found.\n", fullName.c_str());
    }
}

LinuxHwModuleInterface::~LinuxHwModuleInterface()
{
    if (nullptr != hwModule)
    {
        auto const error = dlclose(hwModule);
        if (error)
        {
            Log->Error("FreeLibrary failed!\n");
        }
    }
}

void* LinuxHwModuleInterface::getSymbolAddress(const std::string& symbolName)
{
    return dlsym(hwModule, symbolName.c_str());
}

#endif
