/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "Shape.h"

#include "Expect.h"
#include "GnaException.h"
#include "ModelError.h"

#include "gna2-common-api.h"

#include<algorithm>
#include <vector>

using namespace GNA;

Shape::Shape(ShapeMap && mapIn, gna_tensor_order order) :
    ShapeMap{ std::move(mapIn) },
    LayoutOrder{ order },
    Order{ order }
{}

uint32_t & Shape::operator[](char dimension)
{
    return ShapeMap::operator[](Layout::GetIndex(dimension));
}

uint32_t Shape::at(char dimension) const
{
    return ShapeMap::at(Layout::GetIndex(dimension));
}

Shape Shape::Reshape(gna_tensor_order order) const
{
    const Layout newLayout{ order };
    ShapeMap dims;
    if (GNA_TENSOR_ORDER_ANY == LayoutOrder)
    {
        auto it = LayoutOrder.cbegin();
        for (const auto & dim : newLayout)
        {
            dims[Layout::GetIndex(dim)] = at(*it++);
        }
    }
    else
    {
        for (const auto & dim : newLayout)
        {
            dims[Layout::GetIndex(dim)] = this->at(dim);
        }
    }
    return Shape(std::move(dims), order);
}

Shape Shape::Create(const ApiShape & apiShape, const gna_tensor_order order)
{
    return Shape(Create(std::vector<uint32_t>(apiShape.Dimensions,
        &apiShape.Dimensions[apiShape.NumberOfDimensions]), order), order);
}

ShapeMap Shape::Create(const std::vector<uint32_t> && dimensions, const gna_tensor_order order)
{
    const auto & layout = Layout(order);
    layout.ValidateNumberOfDimensions(dimensions.size());

    ShapeMap shape;
    size_type i = 0;
    for (const auto & dim : dimensions)
    {
        char index = layout.at(i++);
        shape[Layout::GetIndex(index)] = dim;
    }
    return shape;
}

Shape::operator ApiShape() const
{
    ApiShape shape = {};
    shape.NumberOfDimensions = static_cast<uint32_t>(size());
    uint32_t i = 0;
    for (const auto & dim : *this)
    {
        shape.Dimensions[i++] = dim.second;
    }
    return shape;
}

uint32_t Shape::GetNumberOfElements() const
{
    uint32_t counter = 1;
    uint32_t sum = 0;
    for (const auto & dim : *this)
    {
        sum += dim.second;
        if (0 != dim.second)
        {
            counter *= dim.second;
        }
    }
    if (0 == sum)
    {
        return 0;
    }
    return counter;
}

ModelValue Shape::AsModelValue(char dimension, uint32_t operandIndex) const
{
    return ModelValue{ at(dimension), LayoutOrder.GetApiIndex(dimension), operandIndex };
}

void Shape::ExpectFits(const Shape& envelope) const
{
    ProcessEachDimension(envelope, [](auto l, auto r)
    {
        ModelErrorHelper::ExpectBelowEq(l, r, Gna2ItemTypeShapeDimensions);
    });
}

void Shape::ExpectEqual(const Shape& ref) const
{
    ProcessEachDimension(ref, [](auto l, auto r)
    {
        ModelErrorHelper::ExpectEqual(l, r, Gna2ItemTypeShapeDimensions);
    });
}

void Shape::ExpectEqualInverted(const ApiShape & source) const
{
    const auto sourceShape = Create(source, this->Order);
    sourceShape.ExpectEqual(*this);
}

void Shape::ExpectSquare() const
{
    if (size() <= 1) return;
    auto dim1 = this->begin()->second; // removed const as gcc w/a
    auto ctx = ModelItem{ Gna2ItemTypeShapeDimensions };
        ctx.ShapeDimensionIndex = LayoutOrder.GetApiIndex(this->begin()->first);
    std::for_each(begin(), end(),
        [dim1, &ctx](auto const & l)
    {
        ModelErrorHelper::ExpectEqual(l.second, dim1, ctx);
    });
}

bool Shape::IsSquare() const
{
    try
    {
        ExpectSquare();
        return true;
    }
    catch (GnaModelErrorException&)
    {
        return false;
    }
}

void Shape::ProcessEachDimension(const Shape& right, const std::function<void(uint32_t, uint32_t)>& process) const
{
    const auto command = [&](key_type dimension, mapped_type value)
    {
        try
        {
            const auto envelopeVal = right.at(dimension);
            process(value, envelopeVal);
        }
        catch (GnaModelErrorException& e)
        {
            e.SetDimensionIndex(LayoutOrder.GetApiIndex(dimension));
            throw;
        }
    };
    for(const auto& element : *this)
    {
        command(element.first, element.second);
    }
}
