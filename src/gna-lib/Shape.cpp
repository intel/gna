/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "Shape.h"

#include "GnaException.h"
#include "ModelError.h"

#include "gna2-common-api.h"
#include "gna-api-types-xnn.h"

#include <array>
#include <cstddef>
#include <vector>

using namespace GNA;

Shape::Shape(ShapeMap && mapIn, gna_tensor_order order) :
    ShapeMap{ std::move(mapIn) },
    LayoutOrder{ order },
    Order{ order }
{}

Shape::Shape(const gna_3d_dimensions shape) :
    Shape{ GNA_TENSOR_WHD, shape.width, shape.height, shape.depth }
{}

Shape & Shape::operator=(const Shape & right)
{
    ShapeMap::operator=(static_cast<ShapeMap>(right));
    this->Order = right.Order;
    this->LayoutOrder = right.LayoutOrder;
    return (*this);
}

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

Shape::operator gna_3d_dimensions const() const
{
    if (this->count(GNA_DIM_W) > 0 && this->count(GNA_DIM_H) > 0)
    {
        if (this->count(GNA_DIM_D) > 0)
        {
            return gna_3d_dimensions{at(GNA_DIM_W), at(GNA_DIM_H), at(GNA_DIM_D)};
        }
        return gna_3d_dimensions{at(GNA_DIM_W), at(GNA_DIM_H), 0};
    }

    throw GnaException(Gna2StatusXnnErrorLyrInvalidTensorOrder);
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

ModelValue Shape::AsModelValue(char dimension) const
{
    ModelValue mv{ at(dimension) };
    return mv.SetDimension(LayoutOrder.GetApiIndex(dimension));
}

void Shape::ExpectFits(const Shape& envelope) const
{
    ProcessEachDimension(envelope, [](auto l, auto r)
    {
        ModelErrorHelper::ExpectBelowEq(l, r, Gna2ItemTypeShapeDimensions);
    });
}

void Shape::ExpectEqual(const Shape& reference) const
{
    ProcessEachDimension(reference, [](auto l, auto r)
    {
        ModelErrorHelper::ExpectEqual(l, r, Gna2ItemTypeShapeDimensions);
    });
}

void Shape::ExpectEqualInverted(const ApiShape & source) const
{
    const auto sourceShape = Create(source, this->Order);
    sourceShape.ExpectEqual(*this);
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
