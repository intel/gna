/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "gna-api.h"
#include "gna2-model-api.h"
#include "ParameterLimits.h"

#include <map>
#include <string>

namespace GNA
{

class Layout : public std::string
{
public:
    static constexpr size_type MaximumNumberOfDimension = GNA2_SHAPE_MAXIMUM_NUMBER_OF_DIMENSIONS;
    //static const std::vector<gna_tensor_dim> & GetVectorIndices(gna_tensor_order order);
    //static char GetIndex(gna_tensor_dim dim);
    static gna_tensor_dim GetIndex(char dim);

    // Creates default layout as LAYOUT_ANY
    Layout();
    Layout(char const * layoutIn);
    Layout(gna_tensor_order order);
    ~Layout() = default;

    operator gna_tensor_order() const;

    void ValidateNumberOfDimensions(size_type shapeDimensions) const;

    void Reshape(Layout const & newLayout, size_type shapeDimensions);
    int32_t GetApiIndex(gna_tensor_dim dim) const;
    int32_t GetApiIndex(char dim) const;
private:
    static char const * GetOrderString(gna_tensor_order order);

    static const std::map<const std::string, gna_tensor_order> orderStrings;
    //gna_tensor_order OrderFromLayout() const;
};

}
