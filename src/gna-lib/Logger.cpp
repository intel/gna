/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Logger.h"
#include "GnaException.h"
#include "Macros.h"

#include <algorithm>
#include <cstdarg>


using namespace GNA;

auto const & staticDestructionProtectionHelper2 = GNA::StatusHelper::GetStringMap();

Logger::Logger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
    const char * const levelErrorIn) :
    defaultStream{defaultStreamIn},
    component{componentIn},
    levelMessage{levelMessageIn},
    levelError{levelErrorIn}
{}

void Logger::LineBreak() const
{}

void Logger::HorizontalSpacer() const
{}

void Logger::Message(const Gna2Status status) const
{
    UNREFERENCED_PARAMETER(status);
}

void Logger::Message(const Gna2Status status, const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(status);
    UNREFERENCED_PARAMETER(format);
}

void Logger::Message(const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(format);
}

void Logger::Warning(const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(format);
}

void Logger::Error(const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(format);
}

void Logger::Error(const Gna2Status status, const char * const format, ...) const
{
    UNREFERENCED_PARAMETER(status);
    UNREFERENCED_PARAMETER(format);
}

void Logger::Error(const Gna2Status status) const
{
    UNREFERENCED_PARAMETER(status);
}

DebugLogger::DebugLogger(FILE * const defaultStreamIn, const char * const componentIn, const char * const levelMessageIn,
    const char * const levelErrorIn) :
    Logger(defaultStreamIn, componentIn, levelMessageIn, levelErrorIn)
{}

void DebugLogger::LineBreak() const
{
    fprintf(defaultStream, "\n");
}

void DebugLogger::Message(const Gna2Status status) const
{
    printMessage(&status, nullptr, nullptr);
}

void DebugLogger::Message(const Gna2Status status, const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printMessage(&status, format, args);
    va_end(args);
}

void DebugLogger::Message(const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printMessage(nullptr, format, args);
    va_end(args);
}

void DebugLogger::Warning(const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printMessage(nullptr, format, args);
    va_end(args);
}

void DebugLogger::Error(const Gna2Status status) const
{
    printError(&status, nullptr, nullptr);
}

void DebugLogger::Error(const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printError(nullptr, format, args);
    va_end(args);
}

void DebugLogger::Error(const Gna2Status status, const char * const format, ...) const
{
    va_list args;
    va_start(args, format);
    printError(&status, format, args);
    va_end(args);
}

template<typename ... X> void DebugLogger::printMessage(const Gna2Status * const status, const char * const format, X... args) const
{
    printHeader(defaultStream, levelMessage);
    print(defaultStream, status, format, args...);
}

template<typename ... X> void DebugLogger::printWarning(const char * const format, X... args) const
{
    printHeader(defaultStream, levelWarning);
    print(defaultStream, nullptr, format, args...);
}

template<typename ... X> void DebugLogger::printError(const Gna2Status * const status, const char * const format, X... args) const
{
    printHeader(stderr, levelError);
    print(stderr, status, format, args...);
}

inline void DebugLogger::printHeader(FILE * const streamIn, const char * const level) const
{
    try
    {
        fprintf(streamIn, "%s%s", component, level);
    }
    catch (...)
    {
    }
}

template<typename ... X> void DebugLogger::print(FILE * const streamIn, const Gna2Status * const status,
    const char * const format, X... args) const
{
    try
    {
        if (nullptr != status)
        {
        fprintf(streamIn, "Status: %s [%d]\n", StatusHelper::ToString(*status).c_str(), static_cast<int>(*status));
        }
        if (nullptr != format)
        {
            vfprintf(streamIn, format, args...);
        }
        else
        {
            fprintf(streamIn, "\n");
        }
    }
    catch (...)
    {
    }
}

#if defined(DUMP_ENABLED) || DEBUG >= 1
std::unique_ptr<Logger> GNA::Log = std::make_unique<DebugLogger>();
#else // RELEASE
std::unique_ptr<Logger> GNA::Log = std::make_unique<Logger>();
#endif
