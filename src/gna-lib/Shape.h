/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "Layout.h"

#include "ModelError.h"

#include "gna2-model-impl.h"


#include <cstdint>
#include <map>
#include <vector>

namespace GNA
{

using ShapeMap = std::map<gna_tensor_dim, uint32_t>;

struct Shape : public ShapeMap
{
    static Shape Create(const ApiShape & shape, gna_tensor_order order = GNA_TENSOR_ORDER_ANY);

    // Clang issue workaround @see: https://stackoverflow.com/questions/34494765/interaction-between-default-arguments-and-parameter-pack-gcc-and-clang-disagree
    Shape() :
        ShapeMap(),
        Order{ GNA_TENSOR_ORDER_ANY }
    { }

    template<typename ... T>
    Shape(gna_tensor_order order, T ... dimensions) :
        Shape{ Create(std::vector<uint32_t>({ std::forward<T>(static_cast<uint32_t>(dimensions))... }), order), order }
    { }

    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;
    ~Shape() = default;
    ModelValue AsModelValue(char dimension, uint32_t operandIndex = Gna2DisabledU32) const;

    using ShapeMap::at;
    using ShapeMap::operator[];
    uint32_t& operator[](char dimension);
    uint32_t at(char dimension) const;

    operator ApiShape() const;

    Shape Reshape(gna_tensor_order newOrder) const;

    uint32_t GetNumberOfElements() const;

    Layout LayoutOrder;

    // If any dimension is greater than in envelope throws model exception.
    void ExpectFits(const Shape& envelope) const;

    // If any dimension is different than in reference throws model exception.
    void ExpectEqual(const Shape& reference) const;

    void ExpectEqualInverted(const ApiShape & source) const;

    void ExpectSquare() const;

    bool IsSquare() const;

protected:
    static ShapeMap Create(const std::vector<uint32_t> && dimensions,
        gna_tensor_order order = GNA_TENSOR_ORDER_ANY);

    Shape(ShapeMap && map, gna_tensor_order order);

    gna_tensor_order Order;

    void ProcessEachDimension(const Shape& right, const std::function<void(uint32_t, uint32_t)>& process) const;
};
}
