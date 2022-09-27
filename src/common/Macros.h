/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#if defined(__GNUC__)
#define UNREFERENCED_PARAMETER(P) ((void)(P))
#else
#define WIN32_NO_STATUS
#include <windows.h>
#undef WIN32_NO_STATUS
#endif

// Enable safe functions compatibility
#if defined(__STDC_SECURE_LIB__)
#define __STDC_WANT_SECURE_LIB__ 1
#elif defined(__STDC_LIB_EXT1__)
#define STDC_WANT_LIB_EXT1 1
#else
#define memcpy_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
    memcpy(_Destination, _Source, _SourceSize);\
    UNREFERENCED_PARAMETER(_DestinationSize);\
} while(0);
#define memmove_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
    memmove(_Destination, _Source, _SourceSize);\
    UNREFERENCED_PARAMETER(_DestinationSize);\
} while(0);
#define strncpy_s(_Destination, _DestinationSize, _Source, _SourceSize) do {\
    strncpy(_Destination, _Source, _SourceSize);\
    UNREFERENCED_PARAMETER(_DestinationSize);\
} while(0);
#endif

#include <cstring>
