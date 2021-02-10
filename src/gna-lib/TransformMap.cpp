/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "TransformMap.h"

#include "ActivationFunction.h"
#include "AffineFunctions.h"
#include "ConvolutionalFunctions2D.h"
#include "GmmLayer.h"
#include "PoolingFunctions2D.h"
#include "RecurrentFunction.h"
#include "Transform.h"

using namespace GNA;

BaseTransform * TransformList::Emplace(TransformOperation operation,
    const TransformFactoryConfig& config,
    const OperationConfig& operationConfig)
{
    switch (operation)
    {
    case AffineTransform:
    case AffineDiagonalTransform:
        return emplace(AffineFunction::Create(config, operationConfig));
    case RecurrentTransform:
        return emplace(RecurrentFunction::Create(config, operationConfig));
    case ActivationTransform:
        return emplace(ActivationFunction::Create(config));
    case ConvolutionalTransform2D:
        return emplace(ConvolutionFunction2D::Create(config, operationConfig));
    case PoolingTransform2D:
        return emplace(PoolingFunction2D::Create(config, operationConfig));
    case GmmTransform:
        return emplace(GmmFunction::Create(config, operationConfig));
    default:
        throw GnaException(Gna2StatusXnnErrorLyrOperation);
    }
}

// Emplaces transform only if transform is enabled, returns current last transform
BaseTransform * TransformList::emplace(std::unique_ptr<BaseTransform>&& transform)
{
    if (transform)
    {
        // no option for inserting same transform type twice
        if (findTransform(transform->Operation) != __TransformList::cend())
        {
            throw GnaException(Gna2StatusXnnErrorLyrCfg);
        }

        // transform is disabled
        if (!transform)
            return __TransformList::back().get();

        // place transform at the end
        __TransformList::emplace_back(std::move(transform));
    }
    return __TransformList::back().get();
}

__TransformList::const_iterator TransformList::findTransform(TransformOperation transformOperation) const
{
    return std::find_if(__TransformList::cbegin(), __TransformList::cend(),
        [transformOperation](const std::unique_ptr<BaseTransform>& t)
    {
        return t->Operation == transformOperation;
    });
}
