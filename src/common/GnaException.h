/**
 @copyright (C) 2017-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "gna2-common-impl.h"

#include <stdexcept>

namespace GNA
{

/**
 * Custom exception with device open error support
 */
class GnaException : public std::runtime_error
{
public:
    explicit GnaException(Gna2Status status) :
        std::runtime_error{ StatusHelper::ToString(status) },
        Status{ status }
    {}

    Gna2Status GetStatus() const
    {
        return Status;
    }

    virtual ~GnaException() = default;

protected:
    Gna2Status Status;

};

}
