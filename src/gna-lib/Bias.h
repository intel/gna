/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "DataMode.h"
#include "Tensor.h"

#include "ConvolutionKernelArguments.h"

#include <cstdint>

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;
struct Shape;

template<typename T>
struct SetLimits;

struct BiasTensor : public Tensor
{
    BiasTensor(const Shape& dimensions, const uint32_t biasVectorIndex, const DataMode& dataMode,
        void * buffer, const LayerValidator& validator, Gna2BiasMode mode = Gna2BiasModeDefault);

    virtual ~BiasTensor() = default;
    BiasTensor(const Gna2Tensor &apiTensor, uint32_t biasVectorIndex,
                Gna2BiasMode biasMode, const LayerValidator& validatorIn);

    // NOTE: this works only for software mode, HW requires base MB array buffer
    virtual operator const BaseAddress () const override
    {
        return Buffer + (VectorIndex * Mode.Size);
    }

    // NOTE: this works only for software mode, HW requires base MB array buffer
    virtual operator void* () const override
    {
        return Buffer + (VectorIndex * Mode.Size);
    }

    const uint32_t VectorCount;
    const uint32_t VectorIndex;
    const KernelBiasMode BiasMode;

protected:
    static KernelBiasMode ToKernelBiasMode(Gna2BiasMode mode, Gna2TensorMode tensorMode);

    static const FullCapabilitiesMap capabilities;
    static const SetLimits<KernelBiasMode> modeLimits;

private:
    void validate() const;
};


}
