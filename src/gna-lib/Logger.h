/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "gna2-common-api.h"

#include <cstdio>
#include <memory>
#include <string>

namespace GNA
{

// Release build Logger serving mainly as interface
struct Logger
{
    Logger() = default;
    virtual ~Logger() = default;

    virtual void LineBreak() const;
    virtual void HorizontalSpacer() const;

    virtual void Message(const Gna2Status status) const;
    virtual void Message(const Gna2Status status, const char * const format, ...) const;
    virtual void Message(const char * const format, ...) const;

    virtual void Warning(const char * const format, ...) const;

    virtual void Error(const char * const format, ...) const;
    virtual void Error(const Gna2Status status, const char * const format, ...) const;
    virtual void Error(const Gna2Status status) const;

protected:
    Logger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
        const char * const levelErrorIn);

    FILE * const defaultStream = stdout;
    const char * const component = "";
    const char * const levelMessage = "INFO: ";
    const char * const levelWarning = "WARNING: ";
    const char * const levelError = "ERROR: ";

};

// Logger for debug builds
struct DebugLogger : public Logger
{
    DebugLogger() :
        DebugLogger(stderr, "[IntelGna] ", "INFO: ", "ERROR: ")
    {}
    virtual ~DebugLogger() = default;

    virtual void LineBreak() const override;

    virtual void Message(const Gna2Status status) const override;
    virtual void Message(const Gna2Status status, const char * const format, ...) const override;
    virtual void Message(const char * const format, ...) const override;

    virtual void Warning(const char * const format, ...) const override;


    virtual void Error(const Gna2Status status) const override;
    virtual void Error(const char * const format, ...) const override;
    virtual void Error(const Gna2Status status, const char * const format, ...) const override;

protected:
    template<typename ... X> void printMessage(
        const Gna2Status * const status, const char * const format, X... args) const;
    template<typename ... X> void printWarning(const char * const format, X... args) const;
    template<typename ... X>  void printError(
        const Gna2Status * const status, const char * const format, X... args) const;
    inline void printHeader(FILE * const streamIn, const char * const level) const;
    template<typename ... X>  void print(
        FILE * const streamIn, const Gna2Status * const status, const char * const format, X... args) const;

    DebugLogger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
        const char * const levelErrorIn);

};

extern std::unique_ptr<Logger> Log;

}
