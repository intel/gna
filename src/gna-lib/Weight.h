/**
 @copyright Copyright (C) 2018-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "DataMode.h"
#include "Tensor.h"

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;
struct Shape;

struct WeightTensor : public Tensor
{
    WeightTensor(const Shape& dimensions, const DataMode& dataMode,
        void * buffer, const LayerValidator& validator);
    WeightTensor(const Gna2Tensor &apiTensor, const LayerValidator& validator);
    virtual ~WeightTensor() = default;

protected:
    static const FullCapabilitiesMap capabilities;
};

}
