/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "SubModel.h"

using namespace GNA;

SubModel::SubModel(SubmodelType type, uint32_t layerIndex) :
    Type{type},
    LayerIndex{layerIndex},
    layerCount{1}
{}

void SubModel::AddLayer()
{
    layerCount++;
}

uint32_t SubModel::GetLayerCount() const
{
    return layerCount;
}
