/**
 @copyright (C) 2019-2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

enum ThresholdBiasSource
{
    ThresholdBiasSourceDefault,
    ThresholdBiasSourceExternal
};

enum ThresholdInterrupt
{
    ThresholdInterruptDefault,
    ThresholdInterruptNotSent
};

enum ThresholdOperation
{
    ThresholdOperationStop,
    ThresholdOperationContinueIfMet,
    ThresholdOperationContinueIfNotMet,
    ThresholdOperationContinueAllways
};

enum ThresholdCondition
{
    ThresholdConditionScoreNegative,
    ThresholdConditionScoreNotNegative
};
