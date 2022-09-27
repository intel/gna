/**
 @copyright Copyright (C) 2017-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

#include <cstdint>

namespace GNA
{

/******************************************************************************
* GMM Configuration Fields (kept in driver private memory)
******************************************************************************/

/**
* Feature Vector Start Address
*
* Offset:   0x0000
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.1
* Note:     Specifies virtual address pointer to the Feature Vector
*           in user pinned memory
*           Must be 64B aligned
*/
typedef uint32_t FVADDR;

/**
* Feature Vector Offset Configuration
*
* Offset:   0x0004
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.2
* Note:     Specifies the size of each feature vector in bytes
*           including the padding to point to the next feature vector
*           Valid values are:
*               VLENGTH * FVWIDTH -> rounded up to 64B.
*           Only 11 LSbits are used.
*/
typedef uint32_t FVOFFSET;

/**
* Feature Vector Width Configuration
*
* Offset:   0x0008
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.3
* Note:     Specifies feature vector width in bytes
*           Valid values are:
*               0x01 (1 byte [default])
*               others - reserved
*/
typedef uint32_t FVWIDTH;

/**
 * GMM Mode Control register
 *
 * Offset:   0x000C
 * Size:     0x04 B
 * See:      HAS Section 5.4.2.7.4
 * Note:     Controls mode of operation of GMM module
 */
#if defined(_WIN32)
#pragma warning(disable : 201)
#endif
typedef union _GMM_MODE_CTRL
{
    uint32_t     _value;            // value of whole register
    struct _GMM_MODE_BITS
    {
        uint32_t read_elimination : 1;  // 00:00 Const and Var Read Elimination
                                        //      Disable the read of the Const and Var array  and force a use of Const = 0 and VAR = 1
                                        //      0 = normal operation (default)
                                        //      1 = read elimination enabled
        uint32_t calculation_mode : 2;  // 01:02 Calculation mode
                                        //      0 = GMM Mode = L2 euclidean Distance (default)
                                        //      1 = L1 Distance (max(abs()))
                                        //      2 = Linf Manhattan Distance
        uint32_t __res_03 : 29;         // 03:31 Reserved
    };
} GMM_MODE_CTRL;

/**
 * Number of Feature Vector Configuration
 *
 * Offset:   0x0010
 * Size:     0x04 B
 * See:      HAS Section 5.4.2.7.4
 * Note:     Specifies number of feature vectors to score
 *           Valid values are:
 *               from:   1
 *               to:     8
 *               others - reserved
 *           Default is 1
 */
typedef uint32_t NUMFV;

/**
* Vector Length Configuration
*
* Offset:   0x0014
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.5
* Note:     Specifies feature vector length
*           Valid values are:
*               {24, 32, 40, 48, 56, 72, 80, 88, 96}.
*           Default is 0x18
*           Only 8 LSbits are used.
*/
typedef uint32_t VLENGTH;

/**
* Mean Vector Start Address Lower 32-bits
*
* Offset:   0x0018
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.6
* Note:     Specifies virtual address to the Mean Vector in user pinned memory
*           Must be 8B aligned
*/
typedef uint32_t MVADDR;

/**
* Mean Vector Width Configuration
*
* Offset:   0x0020
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.7
* Note:     Specifies mean vector width in bytes.
*           Only 0x01 value allowed (1 byte [default])
*           others - reserved
*           Only 3 LSbits are used.
*/
typedef uint32_t MVWIDTH;

/**
* Mean Vector Set Offset Configuration
*
* Offset:   0x0028
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.8
* Note:     Specifies mean vector set offset in bytes.
*           Valid values are:
*               from:   0x18
*               to:     0x60000 (4096*96*1)
*           Must be multiple of 8
*           Default is 0x18
*           Only 19 LSbits are used.
*/
typedef uint32_t MVSOFFSET;

/**
* Variance Vector Start Address Lower 32-bits
*
* Offset:   0x0030
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.9
* Note:     Specifies virtual address to the Variance Vector
*           in user pinned memory
*           Must be 8B aligned
*/
typedef uint32_t VVADDR;

/**
* Variance Vector Width Configuration
*
* Offset:   0x0038
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.10
* Note:     Specifies variance vector width in bytes.
*           Valid values are:
*               0x001 - 1 byte
*               0x010 - 2 bytes
*               others - reserved
*/
typedef uint32_t VVWIDTH;

/**
* Variance Vector Set Offset Configuration
*
* Offset:   0x0040
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.11
* Note:     Specifies variance vector set offset in bytes.
*           Valid values are:
*               from:   0x18
*               to:     0xC0000 (4096*96*2)
*           Default is 0x18
*           Must be multiple of 8
*           Only 20 LSbits are used.
*/
typedef uint32_t VVSOFFSET;

/**
* Gaussian Constant Start Address Lower 23-bits
*
* Offset:   0x0044
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.12
* Note:     Specifies Gaussian Constant Start virtual address
*           in user pinned memory
*           Must be 8B aligned
*/
typedef uint32_t GCADDR;

/**
* Gaussian Constant Width Configuration
* Variance Vector Width Configuration
*
* Offset:   0x004c
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.13
* Note:     Specifies gaussian constant width in bytes.
*           Valid values are:
*               0x100 - 4 bytes
*               others - reserved
*           Only 3 LSbits are used.
*/
typedef uint32_t GCWIDTH;

/**
* Gaussian Constant Set Offset Configuration
*
* Offset:   0x0050
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.14
* Note:     Specifies the number of bytes from the start of one Gaussian
*           Constant set to the start of the next Gaussian Constant set
*           Valid value are:
*               from:   0x08
*               to:     0x4000 (4096*4).
*           Padding must be added when number of mixtures is odd.
*           Default is 0x08
*           Only 20 LSbits are used
*/
typedef uint32_t GCSOFFSET;

/**
* Maximum Score
*
* Offset:   0x0054
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.15
* Note:     Specifies vendor dependent score threshold representing maximum
*           score (corresponding to smallest log likelihood)
*           SW must initialize this to the appropriate max score result
*/
typedef uint32_t MAXLSSCORE;

/**
* Maximum Score Width Configuration
*
* Offset:   0x0054
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.16
* Note:     Specifies max score width
*           Valid value are:
*               0x100 - 4 bytes (default)
*               others reserved
*           Only 3 LSbits are used
*/
typedef uint32_t MAXLSWIDTH;

/**
* Number of Mixture Components per GMM
*
* Offset:   0x005C
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.17
* Note:     Specifies the number of mixture components per GMM/state
*           Valid value are:
*               from:   0x01
*               to:     0x1000 (4096).
*           Default is 0x01
*           Only 13 LSbits are used
*/
typedef uint32_t NUMMCPG;

/**
* GMM Total Elements in State
*
* Offset:   0x0060
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.18
* Note:     The value in this register must match the total amount of data
*           elements that are used in the mean and var arrays, the size of
*           the element is calculated directly
*           Valid value is:
*               Number of Mixture Components * Feature vectore length
*           Only 19 LSbits are used
*/
typedef uint32_t GMMTELST;
/**
* Number of GMMs/States.
*
* Offset:   0x0064
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.19
* Note:     Specifies the number of GMMs per frame
*           Valid value are:
*               from:   0x01
*               to:     0x40000 (262144).
*           Default is 0x01
*           Only 19 LSbits are used
*/
typedef uint32_t NUMGMMS;

/**
* Active GMM/State List Start Address
*
* Offset:   0x0068
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.20
* Note:     Specifies virtual address pointer to the Active GMM List
*           in user pinned memory
*           Must be 64B aligned
*/
typedef uint32_t ASLADDR;

/**
* Active GMM/State List Length
*
* Offset:   0x0070
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.21
* Note:     Specifies the total number items in the active GMM list
*           This field is used only when in active list mode
*           Valid value are:
*               from:   0x01
*               to:     0x40000 (262144).
*           Default is 0x01
*           Only 19 LSbits are used
*/
typedef uint32_t ASTLISTLEN;

/**
* GMM Score Width (in bytes).
*
* Offset:   0x0074
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.22
* Note:     Specifies GMM score width
*           Valid value are:
*               0x100 - 4 bytes (default)
*               others reserved
*           Only 3 LSbits are used
*/
typedef uint32_t GMMSCRWIDTH;

/**
* GMM Score Start Address
*
* Offset:   0x0078
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.23
* Note:     Specifies virtual address pointer to the region (in user pinned
*           memory) where the GMM Scoring Accelerator should store the results
*           Must be 64B aligned
*/
typedef uint32_t GMMSCRADD;

/**
* GMM Score Length
*
* Offset:   0x007C
* Size:     0x04 B
* See:      HAS Section 5.4.2.7.24
* Note:     The value in this register must match the total amount of data,
*           in bytes, that is stored to the Gscore output array.
*           Valid values are:
*               GMMSCRWIDTH (4) * NUMFV * NumberGMMstates, w/o Active list
*               GMMSCRWIDTH (4) * NUMFV * MaxAct, in active list mode
*           Only 24 LSbits are used
*/
typedef uint32_t GMMSCRLEN;

/******************************************************************************
* xNN Configuration Fields (kept in driver private memory)
******************************************************************************/

/**
* xNN Layer Array Base Address
*
* Offset:   0x0100
* Size:     0x04 B
* See:      HAS Section 5.4.3.8.1
* Note:     Specifies virtual address pointer to the beginning of Layer Array
*           in user pinned memory.
*           Must be 128B aligned
*/
typedef uint32_t LABASE;

/**
* xNN Layer Array Count
*
* Offset:   0x0104
* Size:     0x02 B
* See:      HAS Section 5.4.3.8.2
* Note:     Specifies the number of layers in the xNN array.
*           Valid values are:
*               from:   0x00  - indicates empty array
*               to:     0x400 (1024)
*               others - reserved
*           Default is 0
*           Only 10 LSbits are used
*/
typedef uint16_t LACNT;

}
