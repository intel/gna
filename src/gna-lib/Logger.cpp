/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Logger.h"
#include "GnaException.h"
#include "Macros.h"

#include <algorithm>
#include <stdarg.h>


using namespace GNA;

const std::map<Gna2Status, std::string>& GNA::StatusHelper::GetDescriptionMap()
{
    static const std::map<Gna2Status, std::string> Gna2StatusToStringMap
    {
        { Gna2StatusSuccess, " - Success: Operation successful, no errors or warnings" },
        { Gna2StatusWarningDeviceBusy, " - Warning: Device busy - accelerator is still running, can not enqueue more requests" },
        { Gna2StatusWarningArithmeticSaturation,  " - Warning: Scoring saturation - an arithmetic operation has resulted in saturation" },
        { Gna2StatusUnknownError, " - Unknown error occurred" },
        { Gna2StatusNotImplemented, " - Functionality not implemented yet" },
        { Gna2StatusIdentifierInvalid, " - Identifier is invalid" },
        { Gna2StatusNullArgumentNotAllowed, " - NULL argument not allowed" },
        { Gna2StatusNullArgumentRequired, " - NULL argument is required" },
        { Gna2StatusResourceAllocationError, " - Resources allocation error" },
        { Gna2StatusDeviceNotAvailable, " - Device: not available" },
        { Gna2StatusDeviceNumberOfThreadsInvalid, " - Device: failed to open, thread count is invalid" },
        { Gna2StatusDeviceVersionInvalid, " - Device: version of device invalid" },
        { Gna2StatusDeviceQueueError, " - Device: request queue error" },
        { Gna2StatusDeviceIngoingCommunicationError, " - Device: IOCTL result retrieval failed" },
        { Gna2StatusDeviceOutgoingCommunicationError, " - Device: sending IOCTL failed" },
        { Gna2StatusDeviceParameterOutOfRange, " - Device: Parameter out of Range error occurred" },
        { Gna2StatusDeviceVaOutOfRange, " - Device: Virtual Address out of range on DMA ch." },
        { Gna2StatusDeviceUnexpectedCompletion, " - Device: Unexpected completion during PCIe operation" },
        { Gna2StatusDeviceDmaRequestError, " - Device: DMA error during PCIe operation" },
        { Gna2StatusDeviceMmuRequestError, " - Device: MMU error during PCIe operation" },
        { Gna2StatusDeviceBreakPointHit, " - Device: accelerator paused on breakpoint" },
        { Gna2StatusDeviceCriticalFailure, " - Critical device error occurred, device has been reset" },
        { Gna2StatusMemoryAlignmentInvalid, " - Memory alignment is invalid" },
        { Gna2StatusMemorySizeInvalid, " - Memory size not supported" },
        { Gna2StatusMemoryTotalSizeExceeded, " - Request's model configuration exceeded supported mode limits" },
        { Gna2StatusMemoryBufferInvalid, " - Memory buffer is invalid" },
        { Gna2StatusRequestWaitError, " - Request wait failed" },
        { Gna2StatusActiveListIndicesInvalid, " - Active list indices are invalid" },
        { Gna2StatusAccelerationModeNotSupported,  " - Acceleration mode not supported" },
        { Gna2StatusModelConfigurationInvalid, " - Error: Model configuration is not supported" },
        { Gna2StatusNotMultipleOf, " - Value is not multiple of required value" },
        { Gna2StatusBadFeatLength, "Gna2StatusBadFeatLength" },
        { Gna2StatusDataModeInvalid, " - Data mode value is invalid" },
        { Gna2StatusXnnErrorNetLyrNo, "Gna2StatusXnnErrorNetLyrNo" },
        { Gna2StatusXnnErrorNetworkInputs, "Gna2StatusXnnErrorNetworkInputs" },
        { Gna2StatusXnnErrorNetworkOutputs, "Gna2StatusXnnErrorNetworkOutputs" },
        { Gna2StatusXnnErrorLyrOperation, "Gna2StatusXnnErrorLyrOperation" },
        { Gna2StatusXnnErrorLyrCfg, "Gna2StatusXnnErrorLyrCfg" },
        { Gna2StatusXnnErrorLyrInvalidTensorOrder, "Gna2StatusXnnErrorLyrInvalidTensorOrder" },
        { Gna2StatusXnnErrorLyrInvalidTensorDimensions, "Gna2StatusXnnErrorLyrInvalidTensorDimensions" },
        { Gna2StatusXnnErrorInvalidBuffer, "Gna2StatusXnnErrorInvalidBuffer" },
        { Gna2StatusXnnErrorNoFeedback, "Gna2StatusXnnErrorNoFeedback" },
        { Gna2StatusXnnErrorNoLayers, "Gna2StatusXnnErrorNoLayers" },
        { Gna2StatusXnnErrorGrouping, "Gna2StatusXnnErrorGrouping" },
        { Gna2StatusXnnErrorInputBytes, "Gna2StatusXnnErrorInputBytes" },
        { Gna2StatusXnnErrorInputVolume, "Gna2StatusXnnErrorInputVolume" },
        { Gna2StatusXnnErrorOutputVolume, "Gna2StatusXnnErrorOutputVolume" },
        { Gna2StatusXnnErrorIntOutputBytes, "Gna2StatusXnnErrorIntOutputBytes" },
        { Gna2StatusXnnErrorOutputBytes, "Gna2StatusXnnErrorOutputBytes" },
        { Gna2StatusXnnErrorWeightBytes, "Gna2StatusXnnErrorWeightBytes" },
        { Gna2StatusXnnErrorWeightVolume, "Gna2StatusXnnErrorWeightVolume" },
        { Gna2StatusXnnErrorBiasBytes, "Gna2StatusXnnErrorBiasBytes" },
        { Gna2StatusXnnErrorBiasVolume, "Gna2StatusXnnErrorBiasVolume" },
        { Gna2StatusXnnErrorBiasMode, "Gna2StatusXnnErrorBiasMode" },
        { Gna2StatusXnnErrorBiasMultiplier, "Gna2StatusXnnErrorBiasMultiplier" },
        { Gna2StatusXnnErrorBiasIndex, "Gna2StatusXnnErrorBiasIndex" },
        { Gna2StatusXnnErrorPwlSegments, "Gna2StatusXnnErrorPwlSegments" },
        { Gna2StatusXnnErrorPwlData, "Gna2StatusXnnErrorPwlData" },
        { Gna2StatusXnnErrorConvFltBytes, "Gna2StatusXnnErrorConvFltBytes" },
        { Gna2StatusCnnErrorConvFltCount, "Gna2StatusCnnErrorConvFltCount" },
        { Gna2StatusCnnErrorConvFltVolume, "Gna2StatusCnnErrorConvFltVolume" },
        { Gna2StatusCnnErrorConvFltStride, "Gna2StatusCnnErrorConvFltStride" },
        { Gna2StatusCnnErrorConvFltPadding, "Gna2StatusCnnErrorConvFltPadding" },
        { Gna2StatusCnnErrorPoolStride, "Gna2StatusCnnErrorPoolStride" },
        { Gna2StatusCnnErrorPoolSize, "Gna2StatusCnnErrorPoolSize" },
        { Gna2StatusCnnErrorPoolType, "Gna2StatusCnnErrorPoolType" },
        { Gna2StatusGmmBadMeanWidth, "Gna2StatusGmmBadMeanWidth" },
        { Gna2StatusGmmBadMeanOffset, "Gna2StatusGmmBadMeanOffset" },
        { Gna2StatusGmmBadMeanSetoff, "Gna2StatusGmmBadMeanSetoff" },
        { Gna2StatusGmmBadMeanAlign, "Gna2StatusGmmBadMeanAlign" },
        { Gna2StatusGmmBadVarWidth, "Gna2StatusGmmBadVarWidth" },
        { Gna2StatusGmmBadVarOffset, "Gna2StatusGmmBadVarOffset" },
        { Gna2StatusGmmBadVarSetoff, "Gna2StatusGmmBadVarSetoff" },
        { Gna2StatusGmmBadVarsAlign, "Gna2StatusGmmBadVarsAlign" },
        { Gna2StatusGmmBadGconstOffset, "Gna2StatusGmmBadGconstOffset" },
        { Gna2StatusGmmBadGconstAlign, "Gna2StatusGmmBadGconstAlign" },
        { Gna2StatusGmmBadMixCnum, "Gna2StatusGmmBadMixCnum" },
        { Gna2StatusGmmBadNumGmm, "Gna2StatusGmmBadNumGmm" },
        { Gna2StatusGmmBadMode, "Gna2StatusGmmBadMode" },
        { Gna2StatusGmmCfgInvalidLayout, "Gna2StatusGmmCfgInvalidLayout" },
        { Gna2StatusDriverQoSTimeoutExceeded, " - Request was aborted due to QoS timeout" },
        { Gna2StatusHardwareModuleNotFound, " - Hardware module shared library matching current device not found" },
        { Gna2StatusHardwareModuleSymbolNotFound, " - Hardware module shared library matching current device does not contain required symbol" },
        { Gna2StatusDriverCommunicationMemoryMapError, " - Unsuccessful mapping of the memory" },
    };
    return Gna2StatusToStringMap;
}

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
    fprintf(streamIn, "%s%s", component, level);
}

template<typename ... X> void DebugLogger::print(FILE * const streamIn, const Gna2Status * const status,
    const char * const format, X... args) const
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

#if defined(DUMP_ENABLED) || DEBUG >= 1
std::unique_ptr<Logger> GNA::Log = std::make_unique<DebugLogger>();
#else // RELEASE
std::unique_ptr<Logger> GNA::Log = std::make_unique<Logger>();
#endif
