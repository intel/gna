/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef __GNA2_API_WRAPPER_H
#define __GNA2_API_WRAPPER_H

#include "gna2-common-impl.h"

#include "GnaException.h"
#include "Logger.h"

#include <functional>

namespace GNA
{

class ApiWrapper
{
public:

    template<typename ReturnType = ApiStatus, typename ... ErrorValueType>
    static ReturnType ExecuteSafely(const std::function<ReturnType()>& command,
        ErrorValueType... error)
    {
        try
        {
            return command();
        }
        catch (const std::exception& e)
        {
            LogException(e);
            return ReturnError<ReturnType>(error...);
        }
    }

private:
    template<typename ReturnType, typename... ErrorValueType>
    static ReturnType ReturnError(ErrorValueType... returnValues)
    {
        ReturnType returnValueContainer[1] = {returnValues...};
        return returnValueContainer[0];
    }

    template<typename ExceptionType>
    static void LogException(const ExceptionType& e);
};

template<>
inline ApiStatus ApiWrapper::ReturnError()
{
    return Gna2StatusUnknownError;
}

template<>
inline void ApiWrapper::ReturnError()
{
    return;
}

template<>
inline void ApiWrapper::LogException(const std::exception& e)
{
    Log->Error("Unknown error: %s.\n", e.what());
}

template<>
inline void ApiWrapper::LogException(const GnaException& e)
{
    Log->Error(e.GetStatus(), " GnaException");
}

template<>
inline Gna2Status ApiWrapper::ExecuteSafely(const std::function<Gna2Status()>& command)
{
    try
    {
        return command();
    }
    catch (const GnaException &e)
    {
        LogException(e);
        return e.GetStatus();
    }
    catch (const std::exception& e)
    {
        LogException(e);
        return Gna2StatusUnknownError;
    }
}

}

#endif //ifndef __GNA2_API_WRAPPER_H
