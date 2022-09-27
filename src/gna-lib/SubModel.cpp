/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include "SubModel.h"

#include <algorithm>

using namespace GNA;

SubModel::SubModel(SubModelType type, uint32_t layerIndex) :
    Type{type},
    LayerIndex{layerIndex},
    layerCount{1}
{}

void SubModel::AddLayer()
{
    layerCount++;
}

bool SubModel::IsSoftwareLayer(uint32_t layerIndex,
    const std::vector<std::unique_ptr<SubModel>>& subModels)
{
    return std::any_of(subModels.begin(), subModels.end(),
        [layerIndex](auto && subModel)
    {
        return subModel->Contains(layerIndex) && subModel->Type == Software;
    });
}

uint32_t SubModel::GetLayerCount() const
{
    return layerCount;
}
