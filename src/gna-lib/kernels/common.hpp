/**
 @copyright Copyright (C) 2021 Intel Corporation
 SPDX-License-Identifier: LGPL-2.1-or-later
*/

#include <cstdint>

/** Interface for index sources */
class IndexSource
{
public:
    virtual ~IndexSource() = default;

    /** Check if next index exists */
    virtual bool HasNext() const = 0;

    /** Get next index */
    virtual uint32_t Next() = 0;
};

/** Source that provides indexes in a sequence up to a given limit */
class SequenceIndexSource : public IndexSource
{
public:
    SequenceIndexSource(uint32_t indexLimit)
        : limit{indexLimit}, current{0}
    {
    }

    bool HasNext() const override
    {
        return (current < limit);
    }

    uint32_t Next() override
    {
        return current++;
    }

private:
    uint32_t limit;
    uint32_t current;
};

/** Source that provides indexes from active list */
class ActiveListIndexSource : public IndexSource
{
public:
    ActiveListIndexSource(const AffineConfigAl &affineConfigAl)
        : al{affineConfigAl}, current{0}
    {
    }

    bool HasNext() const override
    {
        return (current < al.count);
    }

    uint32_t Next() override
    {
        return al.indices[current++];
    }

private:
    const AffineConfigAl &al;
    uint32_t current;
};
