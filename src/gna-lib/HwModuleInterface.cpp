/**
 @copyright Copyright (C) 2020-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "HwModuleInterface.hpp"

#include "ConvolutionalFunctions.h"
#include "ConvolutionalFunctions2D.h"
#include "DataMode.h"
#include "Logger.h"
#include "PoolingFunctions2D.h"

#if 1 == GNA_HW_LIB_ENABLED
#include "GNA_ArchCPkg.h"
#include "GNA_ArchCPkg.configs.h"
#else
	typedef enum { GNA_CFG_DEFLT } GNA3_Cfg_t;
#endif

#include <cstdint>
#include <memory>

using namespace GNA;

HwUarchParams::HwUarchParams(struct GNA3_AdaptHW const& source)
#if 1 == GNA_HW_LIB_ENABLED
    :
    Valid{source.Valid},
    KWG{source.KWG},
    KWGIter{source.KWGIter},
    uT{source.uT},
    KMemBase{source.KMemBase},
    CMemBase{source.CMemBase},
    PMemBase{source.PMemBase}
{
}
#else
{
    UNREFERENCED_PARAMETER(source);
}
#endif

HwModuleInterface::HwModuleInterface(DeviceVersion deviceVersion)
{
    if (Gna2DeviceVersion3_0 > deviceVersion || Gna2DeviceVersionEmbedded1_0 == deviceVersion)
    {
        return;
    }
#if 1 == GNA_HW_LIB_ENABLED
    isModuleLoaded = true;
    Log->Message("HwModule library loaded successfully.\n");
#else
    isModuleLoaded = false;
    Log->Warning("HwModule library not loaded.\n");
#endif

    SetConfig(deviceVersion);
}

HwUarchParams HwModuleInterface::GetCnnParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                              const DataMode& outputMode, bool is1D) const
{
    Expect::True(isModuleLoaded, Gna2StatusHardwareModuleNotFound);
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

bool HwModuleInterface::SetConfig(DeviceVersion deviceVersion)
{
    UNREFERENCED_PARAMETER(deviceVersion);
#if 1 == GNA_HW_LIB_ENABLED
    auto config = GNA3_Config_t{};
    GNA3_GetConfig(&config);
    return GNA3_SetConfig(&config);
#else
    return false;
#endif
}

HwUarchParams HwModuleInterface::Get1DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                             const DataMode& outputMode) const
{
#if 1 == GNA_HW_LIB_ENABLED
    auto const LD_2DCNN = GNA3_NewLD();

    if (!LD_2DCNN) {
        throw GnaException(Gna2StatusResourceAllocationError);
    }

    LD_2DCNN->IFV.N = 1; // Must set to 1, for IFVs
    LD_2DCNN->IFV.W = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_W));
    LD_2DCNN->IFV.H = 1;
    LD_2DCNN->IFV.C = 1;
    LD_2DCNN->IFV.Prec = static_cast<GNA3_Prec_t>(cnnIn->Input->Mode.Size);
    // Kernels @ Setting 2DCNNc Parameters
    LD_2DCNN->Op = GNA3_OP_1DCNN;
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.NConvFilters = static_cast<uint16_t>(cnnIn->Filters->Count);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.KPrec = static_cast<GNA3_Prec_t>(cnnIn->Filters->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.NConvFilterElements = static_cast<uint16_t>(cnnIn->Filters->at(GNA_DIM_W));
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.InputConvStride = static_cast<uint16_t>(cnnIn->Stride->at(GNA_DIM_W));
    // BIAS @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.BPrec = static_cast<GNA3_Prec_t>(cnnIn->Biases->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.BType = GNA3_BIASperKERNEL; // other modes not supported

    // Pooling @ Setting 2DCNNc Parameters
    if (poolingIn) {
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PType = static_cast<GNA3_PoolType_t>(GetPoolingMode(poolingIn));
    }
    else {
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PType = GNA3_POOL_DIS;
    }

    if (LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PType != GNA3_POOL_DIS)
    {
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PWin = static_cast<uint8_t>(poolingIn->Window->at(GNA_DIM_W));
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PStr = static_cast<uint8_t>(poolingIn->Stride->at(GNA_DIM_W));
    }
    else
    {
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PWin = 0;
        LD_2DCNN->OpStruct.GNA3_OP_1DCNN.PStr = 0;
    }
    // Activation @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.ACTx = static_cast<GNA3_Prec_t>(outputMode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_1DCNN.NSegs = 0;

    auto const validationResult = GNA3_PopLD(LD_2DCNN);
    auto adaptHW = GNA3_AdaptHW_t{LD_2DCNN->AdaptHW};
    GNA3_FreeLD(LD_2DCNN);

    if (!validationResult)
    {
        adaptHW.Valid = false;
    }

    return HwUarchParams{adaptHW};
#else
    UNREFERENCED_PARAMETER(cnnIn);
    UNREFERENCED_PARAMETER(poolingIn);
    UNREFERENCED_PARAMETER(outputMode);
    throw GnaException(Gna2StatusNotImplemented);
#endif
}

HwUarchParams HwModuleInterface::Get2DParams(ConvolutionFunction2D const* cnnIn, PoolingFunction2D const* poolingIn,
                                             const DataMode& outputMode) const
{
#if 1 == GNA_HW_LIB_ENABLED
    auto const LD_2DCNN = GNA3_NewLD();

    if (!LD_2DCNN) {
        throw GnaException(Gna2StatusResourceAllocationError);
    }

    LD_2DCNN->IFV.N = 1; // Must set to 1, for IFVs
    LD_2DCNN->IFV.H = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_H));
    LD_2DCNN->IFV.W = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_W));
    LD_2DCNN->IFV.C = static_cast<uint16_t>(cnnIn->Input->at(GNA_DIM_D));
    LD_2DCNN->IFV.Prec = static_cast<GNA3_Prec_t>(cnnIn->Input->Mode.Size);
    // Kernels @ Setting 2DCNNc Parameters
    LD_2DCNN->Op = GNA3_OP_2DCNNc;
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KNum = static_cast<uint16_t>(cnnIn->Filters->Count);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KPrec = static_cast<GNA3_Prec_t>(cnnIn->Filters->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KDim.H = static_cast<uint16_t>(cnnIn->Filters->at(GNA_DIM_H));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.KDim.W = static_cast<uint16_t>(cnnIn->Filters->at(GNA_DIM_W));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CStr.H = static_cast<uint16_t>(cnnIn->Stride->at(GNA_DIM_H));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CStr.W = static_cast<uint16_t>(cnnIn->Stride->at(GNA_DIM_W));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CZPad.H = static_cast<uint16_t>(cnnIn->Padding->at(GNA_DIM_H));
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.CZPad.W = static_cast<uint16_t>(cnnIn->Padding->at(GNA_DIM_W));
    // BIAS @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.BPrec = static_cast<GNA3_Prec_t>(cnnIn->Biases->Mode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.BType = GNA3_BIASperKERNEL; // other modes not supported

    // Pooling @ Setting 2DCNNc Parameters
    if (poolingIn) {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PType = static_cast<GNA3_PoolType_t>(GetPoolingMode(poolingIn));
    }
    else {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PType = GNA3_POOL_DIS;
    }

    if (LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PType != GNA3_POOL_DIS)
    {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.H = static_cast<uint16_t>(poolingIn->Window->at(GNA_DIM_H));
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.W = static_cast<uint16_t>(poolingIn->Window->at(GNA_DIM_W));
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.H = static_cast<uint16_t>(poolingIn->Stride->at(GNA_DIM_H));
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.W = static_cast<uint16_t>(poolingIn->Stride->at(GNA_DIM_W));
    }
    else
    {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.H = 0;
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PWin.W = 0;
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.H = 0;
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.PStr.W = 0;
    }
    // Activation @ Setting 2DCNNc Parameters
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.ACTx = static_cast<GNA3_Prec_t>(outputMode.Size);
    LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.NSegs = 0;

    if ((LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.ACTx == GNA3_INT32) && (LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.NSegs == 0)) {
        LD_2DCNN->OpStruct.GNA3_OP_2DCNNc.ACTx = GNA3_DIS;
    }

    auto const validationResult = GNA3_PopLD(LD_2DCNN);
    auto adaptHW = GNA3_AdaptHW_t{LD_2DCNN->AdaptHW};
    GNA3_FreeLD(LD_2DCNN);

    if (!validationResult)
    {
        adaptHW.Valid = false;
    }

    return HwUarchParams{adaptHW};
#else
    UNREFERENCED_PARAMETER(cnnIn);
    UNREFERENCED_PARAMETER(poolingIn);
    UNREFERENCED_PARAMETER(outputMode);
    throw GnaException(Gna2StatusNotImplemented);
#endif
}

bool HwModuleInterface::IsModuleLoaded() const
{
    return isModuleLoaded;
}
