/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "GnaException.h"

#include "gna2-model-impl.h"
#include "Layout.h"

#include <map>
#include <vector>
#include <array>

namespace GNA
{

using StaticCaps = std::array<uint32_t, 3>;

struct ModelErrorSource
{
    constexpr ModelErrorSource(Gna2Status status) :
        legacyStatus(status)
    {}

    Gna2Status legacyStatus;
    operator Gna2Status () const
    {
        return legacyStatus;
    }
};

template<typename T>
struct ValueLimits
{
    T Value = {};
    ModelErrorSource Error = Gna2StatusNotImplemented;
};

using AlignLimits = ValueLimits<uint32_t>;

struct BufferLimits : public AlignLimits
{
    BufferLimits(const uint32_t alignment, const ModelErrorSource error) :
        ValueLimits{alignment, error}
    {}
};

template<typename T>
struct SetLimits : public std::vector<T>
{
    SetLimits(const std::vector<T>& validValuesSet, ModelErrorSource error) :
        std::vector<T>{validValuesSet},
        Error{error}
    {}

    template<typename ... X>
    SetLimits(ModelErrorSource error, X ... limits) :
        std::vector<T>{limits...},
        Error{error}
    {}

    ModelErrorSource Error;
};

using MultiplierMap = std::array<uint32_t, Gna2DataTypeWeightScaleFactor + 1>;

struct MultiplierLimits : protected MultiplierMap
{
    constexpr MultiplierLimits() :
        MultiplierMap{},
        Error{Gna2StatusNotImplemented}
    {}

    constexpr MultiplierLimits(const MultiplierMap& multipliers, ModelErrorSource error) :
        MultiplierMap{multipliers},
        Error{error}
    {
    }

    MultiplierLimits(const MultiplierLimits& multipliers, ModelErrorSource error);

    MultiplierLimits(uint32_t multiplier, ModelErrorSource error);

    uint32_t& at(Gna2DataType type);

    uint32_t at(Gna2DataType type) const;

    void SetEffective(DataType type);

    uint32_t GetEffective() const;

    ModelErrorSource Error;
};

template<typename T = uint32_t>
struct RangeLimits
{
    constexpr RangeLimits() = default;

    constexpr RangeLimits(T min, ModelErrorSource minError, T max, ModelErrorSource maxError, const MultiplierLimits& multipliers) :
        Min{min, minError},
        Max{max, maxError},
        Multipliers{multipliers}
    {
    }

    RangeLimits(T min, ModelErrorSource minError, T max, ModelErrorSource maxError, T multiplier, ModelErrorSource multiplierError) :
        RangeLimits{min, minError, max, maxError,
            MultiplierLimits(multiplier, multiplierError)}
    {}

    RangeLimits(T min, T max, T multiplier, ModelErrorSource error) :
        RangeLimits{min, error, max, error, multiplier, error}
    {}

    RangeLimits(T min, T max, ModelErrorSource rangeError, T multiplier, ModelErrorSource multiplierError) :
        RangeLimits{min, rangeError, max, rangeError, multiplier, multiplierError}
    {}

    constexpr RangeLimits(T min, T max, const MultiplierMap& multipliers, ModelErrorSource error) :
        RangeLimits{min, error, max, error, MultiplierLimits(multipliers, error)}
    {}

    constexpr RangeLimits(const StaticCaps& limits, ModelErrorSource error) :
        RangeLimits{limits[0], limits[1], limits[2], error}
    {}

    RangeLimits(T min, T max, const MultiplierLimits& multipliers, ModelErrorSource error) :
        RangeLimits{min, error, max, error, multipliers}
    {}

    RangeLimits(RangeLimits const & base, ModelErrorSource error) :
        RangeLimits{base.Min.Value, error, base.Max.Value, error, MultiplierLimits(base.Multipliers, error)}
    {}

    RangeLimits(const std::vector<uint32_t>::const_iterator& limits, ModelErrorSource error) :
        RangeLimits{limits[0], limits[1], limits[2], error}
    {}

    ValueLimits<T> Min;
    ValueLimits<T> Max;
    // multipliers for different data sizes
    // first (index 0) is effective multiplier (either set by component based on mode or the only one)
    MultiplierLimits Multipliers;
};

using ShapeLimits = std::map<const gna_tensor_dim, RangeLimits<uint32_t>>;

template<Gna2Status error>
auto MakeShapeLimits(const std::vector<uint32_t>& dimensions, gna_tensor_order order)
{
    const auto & layout = Layout(order);
    layout.ValidateNumberOfDimensions(dimensions.size() / 3);

    auto limits = ShapeLimits{};
    auto i = dimensions.begin();
    for (const auto & dim : layout)
    {
        limits[Layout::GetIndex(dim)] = RangeLimits<uint32_t>{ i, error };
        i += 3;
    }
    return limits;
}

struct Shape;

struct OrderLimits : public ValueLimits<gna_tensor_order>
{
    OrderLimits(const gna_tensor_order order) :
        ValueLimits{ order, Gna2StatusXnnErrorLyrInvalidTensorOrder }
    {}
};

struct ComponentLimits
{
    ComponentLimits(const ComponentLimits&) = default;
    ComponentLimits(const OrderLimits order, const ShapeLimits& dimensions) :
        Order{ order },
        Dimensions{ dimensions }
    {}

    OrderLimits Order;
    ShapeLimits Dimensions;
};

}
