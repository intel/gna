/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

//*****************************************************************************
// AccelerationDetector.h - declarations for acceleration modes dispatcher
//                 (currently CPU instruction set extensions only)
// Note:   CPU instruction set extensions detection code based on
//          "Intel® Architecture Instruction Set Extensions Programming reference"
//

#pragma once

#include "XnnKernel.h"

#include "DataMode.h"
#include "Expect.h"
#include "ModelError.h"

#include "gna2-inference-api.h"
#include "gna2-inference-impl.h"

#include <map>
#include <vector>

namespace GNA
{
struct KernelMode
{
    constexpr KernelMode(const DataMode & input, const DataMode & weight, const DataMode & bias) :
        KernelMode{ input.Type, weight.Type, bias.Type }
    {}

    constexpr KernelMode(const DataMode & input) :
        KernelMode{ input.Type }
    {}

    constexpr KernelMode(DataType input, DataType weight, DataType bias) :
        Value{ static_cast<DataType>((input << 16) | (weight << 8) | translateBias(bias)) }
    {}

    constexpr KernelMode(DataType input) :
        KernelMode(input, Gna2DataTypeNone, Gna2DataTypeNone)
    {}

    ~KernelMode() = default;

    constexpr bool operator<(const KernelMode &mode) const
    {
        return mode.Value < Value;
    }

protected:

    /** kernels use single mode for bias type 8/16/32b */
    static constexpr DataType translateBias(DataType bias)
    {
        switch (bias)
        {
        case Gna2DataTypeNone:
        case Gna2DataTypeInt16:
        case Gna2DataTypeInt32:
            return Gna2DataTypeInt8;
        default:
            return bias;
        }
    }

    DataType Value = Gna2DataTypeNone;
};

typedef enum _kernel_op
{
    KERNEL_AFFINE = INTEL_AFFINE,
    KERNEL_AFFINE_DIAGONAL = INTEL_AFFINE_DIAGONAL,
    KERNEL_AFFINE_MULTIBIAS = INTEL_AFFINE_MULTIBIAS,
    KERNEL_CONVOLUTIONAL = INTEL_CONVOLUTIONAL,
    KERNEL_COPY = INTEL_COPY,
    KERNEL_TRANSPOSE = INTEL_DEINTERLEAVE,
    KERNEL_GMM = INTEL_GMM,
    KERNEL_RECURRENT = INTEL_RECURRENT,
    KERNEL_CONVOLUTIONAL_2D = INTEL_CONVOLUTIONAL_2D,
    KERNEL_POOLING_2D = GNA_LAYER_CNN_2D_POOLING,
    KERNEL_POOLING,
    KERNEL_PWL,
    KERNEL_AFFINE_AL,
    KERNEL_GMM_AL,

} kernel_op;

/**
 * Manages runtime acceleration modes
 * and configures execution kernels for given acceleration
 */
class AccelerationDetector
{

public:

    AccelerationDetector();

    ~AccelerationDetector() = default;

    const std::vector<Gna2AccelerationMode>& GetSupportedCpuAccelerations() const;

    template<typename KernelType>
    static const KernelMap<KernelType>&
    GetKernelMap(kernel_op operation, KernelMode dataMode = {Gna2DataTypeInt16})
    {
        try
        {
            return reinterpret_cast<const KernelMap<KernelType>&>(
                GetKernels(operation, dataMode));
        }
        catch (std::out_of_range &)
        {
            throw GnaModelErrorException{ Gna2ItemTypeOperandType, Gna2ErrorTypeNotInSet, 0 };
        }
    }

    void SetHardwareAcceleration(bool isHardwareSupported)
    {
        accelerationModes[AccelerationMode{ Gna2AccelerationModeHardware }] = isHardwareSupported;
    }

    void PrintAllAccelerationModes() const;
protected:
    std::map<AccelerationMode, bool> accelerationModes;

private:
    void DetectSoftwareAccelerationModes();
    //sorted from slowest to fastest
    std::vector<Gna2AccelerationMode> supportedCpuAccelerations;

    static const KernelMap<VoidKernel>& GetKernels(kernel_op operation, KernelMode dataMode);
};

}
