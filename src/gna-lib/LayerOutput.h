/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "Tensor.h"

#include "common.h"

namespace GNA
{
class FullCapabilitiesMap;
class LayerValidator;

struct LayerOutput : public Tensor
{
    LayerOutput(const nn_layer &layer, const LayerValidator& validatorIn);
    LayerOutput(const Gna2Operation &operation, const LayerValidator& validatorIn);
    virtual ~LayerOutput() = default;

    const Tensor ScratchPad;
    const uint32_t Grouping;
    const uint32_t ElementCount;

    ModelValue AsModelValue(char dimension) const override;

protected:
    static const FullCapabilitiesMap capabilities;

    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(
        const Gna2Operation& operation, const LayerValidator& validatorIn) const override;
    virtual std::pair<uint32_t, uint32_t> getGroupingAndElements(const nn_layer& layer) const override;

private:
    static const FullCapabilitiesMap& GetCapabilitiesLegacy();
    static Shape ConvertInCaseOfNewApiOrder(gna_tensor_order getOrder, const uint32_t nOutputColumns, const uint32_t nOutputRows);
    static ApiShape GetShape(const Gna2Operation & operation);
};

}
