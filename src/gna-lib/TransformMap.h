/**
 @copyright (C) 2018-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "GnaException.h"
#include "OperationConfig.h"

#include "common.h"
#include "gna-api-status.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace GNA
{

struct TransformFactoryConfig;
class BaseTransform;

using __TransformList =
    std::vector<std::unique_ptr<BaseTransform>>;

class TransformList : public __TransformList
{
public:
    BaseTransform * Emplace(TransformOperation operation, const TransformFactoryConfig& config,
        const OperationConfig& operationConfig);

    template<typename TransformFunction = BaseTransform>
    TransformFunction * Get(TransformOperation operation) const
    {
        try
        {
            const auto transform = findTransform(operation);
            if (transform != __TransformList::cend())
                return static_cast<TransformFunction *>(transform->get());
        }
        catch (const std::out_of_range&)
        {
        }
        // finally:
        return nullptr;
    }

private:
    // Emplaces transform only if transform is enabled, returns current last transform
    BaseTransform * emplace(std::unique_ptr<BaseTransform>&& transform);

    __TransformList::const_iterator findTransform(TransformOperation transformOperation) const;
};

}
