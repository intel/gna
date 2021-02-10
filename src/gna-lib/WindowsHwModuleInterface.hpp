/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#ifdef WIN32

#include "HwModuleInterface.hpp"

#include <windows.h>

namespace GNA
{
class WindowsHwModuleInterface : public HwModuleInterface
{
public:
    explicit WindowsHwModuleInterface(char const* moduleName);
    ~WindowsHwModuleInterface() override;
    WindowsHwModuleInterface(const WindowsHwModuleInterface&) = delete;
    WindowsHwModuleInterface& operator=(const WindowsHwModuleInterface&) = delete;

private:
    HINSTANCE hwModule;
    void* getSymbolAddress(const std::string& symbolName) override;
};
}

#endif
