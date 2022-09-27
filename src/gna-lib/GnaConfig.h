/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include "GnaTypes.h"

namespace GNA
{

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size

typedef uint8_t     __1B_RES;       // 1B of reserved memory

/******************************************************************************
*
* GNA HW data configuration definitions
*
******************************************************************************/

/**
 * GMM Configuration
 *
 * Offset:  0x0100 (interchangeably with xNN Configuration)
 * Size:    0x80
 * See:     HAS Section 5.4.2.7
 * Note:    Specifies Acoustic Model and scoring parameters
 */
typedef union _GMM_CONFIG
{
    struct _GMM_DESCRIPTOR
    {
        FVADDR      fvaddr;             // 0000 - 0003
        FVOFFSET    fvoffset;           // 0004 - 0007
        FVWIDTH     fvwidth;            // 0008 - 000B
        GMM_MODE_CTRL mode;             // 000C - 000F GMM mode control
        NUMFV       numfv;              // 0010 - 0013
        VLENGTH     vlength;            // 0014 - 0017
        MVADDR      mvaddr;             // 0018 - 001B
        __1B_RES    __res_001c[4];      // 001C - 001F (4B reserved)
        MVWIDTH     mvwidth;            // 0020 - 0023
        __1B_RES    __res_0024[4];      // 0024 - 0027 (4B reserved)
        MVSOFFSET   mvsoffset;          // 0028 - 002B
        __1B_RES    __res_002c[4];      // 002C - 002F (4B reserved)
        VVADDR      vvaddr;             // 0030 - 0033
        __1B_RES    __res_0034[4];      // 0034 - 0037 (4B reserved)
        VVWIDTH     vvwidth;            // 0038 - 003B
        __1B_RES    __res_003c[4];      // 003C - 003F (4B reserved)
        VVSOFFSET   vvsoffset;          // 0040 - 0043
        GCADDR      gcaddr;             // 0044 - 0047
        __1B_RES    __res_0048[4];      // 0048 - 004B (4B reserved)
        GCWIDTH     gcwidth;            // 004C - 004F
        GCSOFFSET   gcsoffset;          // 0050 - 0053
        MAXLSSCORE  maxlsscore;         // 0054 - 0057
        MAXLSWIDTH  maxlswidth;         // 0058 - 005B
        NUMMCPG     nummcpg;            // 005C - 005F
        GMMTELST    gmmtelst;           // 0060 - 0063
        NUMGMMS     numgmms;            // 0064 - 0067
        ASLADDR     asladdr;            // 0068 - 006B
        __1B_RES    __res_006c[4];      // 006C - 006F (4B reserved)
        ASTLISTLEN  astlistlen;         // 0070 - 0073
        GMMSCRWIDTH gmmscrwdth;         // 0074 - 0077
        GMMSCRADD   gmmscradd;          // 0078 - 007B
        GMMSCRLEN   gmmscrlen;          // 007C - 007F
    };
    uint32_t _value[32];// value of whole Configuration

} GMM_CONFIG;                           // GMM Configuration

static_assert(128 == sizeof(GMM_CONFIG), "Invalid size of GMM_CONFIG");

/**
 * xNN Configuration
 *
 * Offset:  0x0100 (interchangeably with GMM Configuration)
 * Size:    0x06
 * See:     HAS Section 5.4.3.8
 * Note:    Specifies Neural Network and scoring parameters
 */
typedef struct _XNN_CONFIG
{
    LABASE      labase;             // 0100 - 0103 - Layer array base
    LACNT       lacnt;              // 0104 - 0105 - Layer array count

} XNN_CONFIG;                       // xNN Configuration

static_assert(6 == sizeof(XNN_CONFIG), "Invalid size of XNN_CONFIG");

/**
* xNN Data Structures - xNN Operation Type
*
* See:     HAS Section 5.4.3.1
* Note:    Enumerates the supported operations by the xNN operation
*/
typedef enum _NN_OP_TYPE : uint8_t
{
    NN_AFFINE = 0x00,
    NN_AFF_AL = 0x01,
    NN_DIAG = 0x02,
    NN_RNN = 0x04,
    NN_CNN = 0x08,
    NN_AFF_MB = 0x09,
    NN_PMA = 0x0A,
    NN_DEINT = 0x10,
    NN_INTER = 0x11,
    NN_COPY = 0x12,
    NN_GMM = 0x20,
    NN_GMM_ACTIVE_LIST = 0x21,
    NN_CNN2D_FUSED = 0x30,
    NN_CNN2D_POOLING = 0x31,
    NN_CNN2D_ADDITION = 0x32,
    NN_CNN2D_CONVERTION = 0x33,
    NN_RESERVED = 0xff

} NN_OP_TYPE;

static_assert(1 == sizeof(NN_OP_TYPE), "Invalid size of NNOOPERATIONTYPE");

# pragma pack ()

}
