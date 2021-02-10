/**
 @copyright (C) 2020-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "ApiWrapper.h"

#include "Expect.h"
#include "StringHelper.h"

#include "gna2-common-impl.h"

#include <map>


#if !defined(_WIN32)
#include <assert.h>
#endif

/**
 Verifies data sizes used in the API and GNA hardware

 @note If data sizes in an application using API differ from data sizes
       in the API library implementation, scoring will not work properly.
 */
static_assert(1 == sizeof(int8_t), "Invalid size of int8_t");
static_assert(2 == sizeof(int16_t), "Invalid size of int16_t");
static_assert(4 == sizeof(int32_t), "Invalid size of int32_t");
static_assert(1 == sizeof(uint8_t), "Invalid size of uint8_t");
static_assert(2 == sizeof(uint16_t), "Invalid size of uint16_t");
static_assert(4 == sizeof(uint32_t), "Invalid size of uint32_t");

const std::map<Gna2Status, std::string>& GNA::StatusHelper::GetStringMap()
{
    static const std::map<Gna2Status, std::string> Gna2StatusToStringMap
    {
        { Gna2StatusSuccess, "Gna2StatusSuccess" },
        { Gna2StatusWarningDeviceBusy, "Gna2StatusWarningDeviceBusy" },
        { Gna2StatusWarningArithmeticSaturation, "Gna2StatusWarningArithmeticSaturation" },
        { Gna2StatusUnknownError, "Gna2StatusUnknownError" },
        { Gna2StatusNotImplemented, "Gna2StatusNotImplemented" },
        { Gna2StatusIdentifierInvalid, "Gna2StatusIdentifierInvalid" },
        { Gna2StatusNullArgumentNotAllowed, "Gna2StatusNullArgumentNotAllowed" },
        { Gna2StatusNullArgumentRequired, "Gna2StatusNullArgumentRequired" },
        { Gna2StatusResourceAllocationError, "Gna2StatusResourceAllocationError" },
        { Gna2StatusDeviceNotAvailable, "Gna2StatusDeviceNotAvailable" },
        { Gna2StatusDeviceNumberOfThreadsInvalid, "Gna2StatusDeviceNumberOfThreadsInvalid" },
        { Gna2StatusDeviceVersionInvalid, "Gna2StatusDeviceVersionInvalid" },
        { Gna2StatusDeviceQueueError, "Gna2StatusDeviceQueueError" },
        { Gna2StatusDeviceIngoingCommunicationError, "Gna2StatusDeviceIngoingCommunicationError" },
        { Gna2StatusDeviceOutgoingCommunicationError, "Gna2StatusDeviceOutgoingCommunicationError" },
        { Gna2StatusDeviceParameterOutOfRange, "Gna2StatusDeviceParameterOutOfRange" },
        { Gna2StatusDeviceVaOutOfRange, "Gna2StatusDeviceVaOutOfRange" },
        { Gna2StatusDeviceUnexpectedCompletion, "Gna2StatusDeviceUnexpectedCompletion" },
        { Gna2StatusDeviceDmaRequestError, "Gna2StatusDeviceDmaRequestError" },
        { Gna2StatusDeviceMmuRequestError, "Gna2StatusDeviceMmuRequestError" },
        { Gna2StatusDeviceBreakPointHit, "Gna2StatusDeviceBreakPointHit" },
        { Gna2StatusDeviceCriticalFailure, "Gna2StatusDeviceCriticalFailure" },
        { Gna2StatusMemoryAlignmentInvalid, "Gna2StatusMemoryAlignmentInvalid" },
        { Gna2StatusMemorySizeInvalid, "Gna2StatusMemorySizeInvalid" },
        { Gna2StatusMemoryTotalSizeExceeded, "Gna2StatusMemoryTotalSizeExceeded" },
        { Gna2StatusMemoryBufferInvalid, "Gna2StatusMemoryBufferInvalid" },
        { Gna2StatusRequestWaitError, "Gna2StatusRequestWaitError" },
        { Gna2StatusActiveListIndicesInvalid, "Gna2StatusActiveListIndicesInvalid" },
        { Gna2StatusAccelerationModeNotSupported, "Gna2StatusAccelerationModeNotSupported" },
        { Gna2StatusModelConfigurationInvalid, "Gna2StatusModelConfigurationInvalid" },
        { Gna2StatusNotMultipleOf, "Gna2StatusNotMultipleOf" },
        { Gna2StatusBadFeatLength, "Gna2StatusBadFeatLength" },
        { Gna2StatusDataModeInvalid, "Gna2StatusDataModeInvalid" },
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
        { Gna2StatusDriverQoSTimeoutExceeded, "Gna2StatusDriverQoSTimeoutExceeded" },
        { Gna2StatusHardwareModuleNotFound, "Gna2StatusHardwareModuleNotFound" },
        { Gna2StatusHardwareModuleSymbolNotFound, "Gna2StatusHardwareModuleSymbolNotFound" },
        { Gna2StatusDriverCommunicationMemoryMapError, "Gna2StatusDriverCommunicationMemoryMapError" },
    };
    return Gna2StatusToStringMap;
}

std::string GNA::StatusHelper::ToString(const Gna2Status statusIn)
{
    const auto& allKnown = GetStringMap();
    const auto found = allKnown.find(statusIn);
    if (found == allKnown.end())
    {
        return std::string{ "Gna2Status[" } +std::to_string(statusIn) + "]";
    }
    return found->second;
}

GNA2_API enum Gna2Status Gna2StatusGetMessage(
    enum Gna2Status status,
    char * messageBuffer,
    uint32_t messageBufferSize)
{
    const std::function<Gna2Status()> command = [&]()
    {
        GNA::Expect::NotNull(messageBuffer);
        const auto& found = GNA::StringHelper::GetFromMap(GNA::StatusHelper::GetStringMap(), status);
        GNA::StringHelper::Copy(*messageBuffer, messageBufferSize, found);
        return Gna2StatusSuccess;
    };
    return GNA::ApiWrapper::ExecuteSafely(command);
}

GNA2_API uint32_t Gna2StatusGetMaxMessageLength()
{
    return GNA::StringHelper::GetMaxLength(GNA::StatusHelper::GetStringMap());
}

Gna2DeviceVersion GNA::Gna2GetVersionForLegacy(gna_device_version legacyVersion)
{
    static const std::unordered_map<gna_device_version, Gna2DeviceVersion, GNA::EnumHash> DeviceVersionMapInverted =
    {
        {GNA_GMM, Gna2DeviceVersionGMM },
        {GNA_0x9, Gna2DeviceVersion0_9 },
        {GNA_1x0, Gna2DeviceVersion1_0 },
        {GNA_2x0, Gna2DeviceVersion2_0 },
        {GNA_EMBEDDED_1x0, Gna2DeviceVersionEmbedded1_0 },
        {GNA_SOFTWARE_EMULATION, Gna2DeviceVersionSoftwareEmulation }
    };
    return DeviceVersionMapInverted.at(legacyVersion);
}
