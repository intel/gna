/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef WIN32

#pragma once

#include "HwModuleInterface.hpp"

namespace GNA
{

class LinuxHwModuleInterface : public GNA::HwModuleInterface
{
public:
    LinuxHwModuleInterface(char const * moduleName);
    virtual ~LinuxHwModuleInterface() override;

private:
    LinuxHwModuleInterface(const LinuxHwModuleInterface &) = delete;
    LinuxHwModuleInterface& operator=(const LinuxHwModuleInterface&) = delete;

    void * hwModule;
    void* getSymbolAddress(const std::string& symbolName) override;
};

}

#endif