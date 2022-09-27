/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Address.h"
#include "Component.h"
#include "DataMode.h"
#include "ParameterLimits.h"
#include "Shape.h"


#include <cstdint>

namespace GNA
{
class Validator;

struct Tensor : public Component
{
    Tensor(const ApiTensor& tensor, uint32_t operandIndex = Gna2DisabledU32);

    Tensor(const ApiTensor& tensor, gna_tensor_order order, const Validator& validator, uint32_t operandIndex = Gna2DisabledU32);

    Tensor(const ApiTensor& apiTensor, const Validator& validatorIn, uint32_t operandIndex = Gna2DisabledU32);

    Tensor(const Shape& dimensions, const DataMode & dataMode, void const * buffer, uint32_t operandIndex = Gna2DisabledU32);

    Tensor(const Shape& dimensions, const DataMode & dataMode,
        void const * buffer, const Validator& validatorIn, uint32_t operandIndex = Gna2DisabledU32);

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

    bool operator == (const std::nullptr_t &right) const
    {
        return Buffer == right;
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
    Tensor(const Tensor& tensor, const Validator& validatorIn, uint32_t operandIndex = Gna2DisabledU32);

    void validate() const;

    uint32_t getGrouping(const Gna2Operation& operation, const LayerValidator& validatorIn) const
    {
        return getGroupingAndElements(operation, validatorIn).first;
    }

    uint32_t getElementCount(const Gna2Operation& operation, const LayerValidator& validatorIn) const
    {
        return getGroupingAndElements(operation, validatorIn).second;
    }

    // Returns pair<grouping, elementCount>
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const;

private:
    static uint32_t getEffectiveSize(const DataMode& mode, uint32_t count);
};

struct TensorLimits : public ComponentLimits
{
    TensorLimits(const ComponentLimits limits, const DataModeLimits& modes) :
        ComponentLimits{ limits },
        Modes{ modes },
        addressAlign{ GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        addressAlign{ GNA_MEM_ALIGN, Gna2StatusMemoryAlignmentInvalid }
    {
    }

    TensorLimits(const OrderLimits order, const ShapeLimits& dimensions, const DataModeLimits& modes,
        const AlignLimits& align) :
        ComponentLimits{ order, dimensions },
        Modes{ modes },
        addressAlign{ align }
    {
    }

    const AlignLimits& GetAddressAlign() const;
    static void OverrideAlign(const uint32_t newAlign);
    const DataModeLimits Modes;
private:
    const AlignLimits addressAlign;
    static const AlignLimits* overridenAlign;
};

template<uint32_t OperandIndex>
struct OperandTensor : public Tensor
{
    OperandTensor(const Shape& dimensions, const DataMode& dataMode,
        void * buffer, const Validator& validatorIn) :
        Tensor{ dimensions, dataMode, buffer, validatorIn, OperandIndex }
    {}

    OperandTensor(const Gna2Tensor &apiTensor, const Validator& validatorIn) :
        Tensor{ apiTensor, validatorIn, OperandIndex }
    {}

    virtual ~OperandTensor() = default;
};

struct OutputTensor : public Tensor
{
    OutputTensor(const Shape& dimensions, const DataMode& dataMode,
        void * buffer, const LayerValidator & validatorIn, const FullCapabilitiesMap & capabilitiesIn) :
        Tensor{ dimensions, dataMode, buffer, Validator{ validatorIn, capabilitiesIn, true }, OutputOperandIndex }
    {}

    virtual ~OutputTensor() = default;
};

using WeightScalesTensor = OperandTensor<WeightScaleFactorOperandIndex>;
using PwlTensor = OperandTensor<PwlOperandIndex>;

}
