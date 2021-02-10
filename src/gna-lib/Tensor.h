/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Address.h"
#include "Component.h"
#include "DataMode.h"
#include "ParameterLimits.h"
#include "Shape.h"

#include "gna-api-status.h"
#include "gna-api-types-xnn.h"

#include <cstdint>

namespace GNA
{
class Validator;

struct Tensor : public Component
{
    Tensor(const ApiTensor& tensor);

    Tensor(const ApiTensor& tensor, gna_tensor_order order, const Validator& validator);

    Tensor(const ApiTensor& apiTensor, const Validator& validatorIn);

    Tensor(const Shape& dimensions, const DataType dataType,
        const TensorMode tensorMode, void const * buffer);

    Tensor(const Shape& dimensions, const DataMode& dataMode,
        void const * buffer, const Validator& validatorIn);

    virtual ~Tensor() = default;

    void UpdateBuffer(const BaseAddress& buffer);

    void ValidateBuffer(const void* const buffer) const;

    virtual operator const BaseAddress() const
    {
        return Buffer;
    }

    virtual operator void* () const
    {
        return Buffer;
    }

    explicit operator ApiTensor() const
    {
        ApiTensor tensor{};
        tensor.Shape = Dimensions;
        tensor.Mode = Mode.Mode;
        tensor.Type = Mode.Type;
        tensor.Data = Buffer;
        if (Layout() != Dimensions.LayoutOrder)
        {
            snprintf(tensor.Layout, sizeof(tensor.Layout), "%s", Dimensions.LayoutOrder.c_str());
        }
        return tensor;
    }

    static DataMode GetDataMode(const Gna2Tensor& tensor);

    const DataMode Mode;

    // Total size in bytes of tensor data buffer
    const uint32_t Size;

    BaseAddress Buffer;

    static Shape GetDimensions(const ApiTensor& operand, gna_tensor_order order)
    {
        return Shape::Create(operand.Shape, order);
    }

protected:
    Tensor(const Tensor& tensor, const Validator& validatorIn);

    void validate() const;

    uint32_t getGrouping(const Gna2Operation& operation, const LayerValidator& validatorIn) const
    {
        return getGroupingAndElements(operation, validatorIn).first;
    }

    uint32_t getGrouping(const nn_layer& layer) const
    {
        return getGroupingAndElements(layer).first;
    }

    uint32_t getElementCount(const Gna2Operation& operation, const LayerValidator& validatorIn) const
    {
        return getGroupingAndElements(operation, validatorIn).second;
    }

    uint32_t getElementCount(const nn_layer& layer) const
    {
        return getGroupingAndElements(layer).second;
    }

    // Returns pair<grouping, elementCount>
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const;
    // Returns pair<grouping, elementCount>
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(const nn_layer& layer) const;

private:
    static uint32_t getEffectiveSize(const DataMode& mode, uint32_t count);

    void validateDimensions() const;
};

struct TensorLimits : public ComponentLimits
{
    TensorLimits(const ComponentLimits limits, const DataModeLimits& modes) :
        ComponentLimits{ limits },
        Modes{ modes },
        Align{ GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        Align{ GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes,
        const AlignLimits& align) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        Align{ align }
    {
    }

    const DataModeLimits Modes;
    const AlignLimits Align;
};

}




