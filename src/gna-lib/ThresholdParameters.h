/**
 @copyright Copyright (C) 2019-2022 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#pragma once

enum ThresholdSource
{
    ThresholdSourceDefault,
    ThresholdSourceExternal
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
    ThresholdOperationContinueAlways
};

enum ThresholdCondition
{
    ThresholdConditionScoreNegative,
    ThresholdConditionScoreNotNegative
};
