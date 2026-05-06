#include <ionic/utils.h>
#include "helpers.h"

static int test_is_power_of_two_zero(void)
{
    if (ionic_is_power_of_two(0) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_one(void)
{
    if (ionic_is_power_of_two(1) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_two(void)
{
    if (ionic_is_power_of_two(2) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_three(void)
{
    if (ionic_is_power_of_two(3) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_four(void)
{
    if (ionic_is_power_of_two(4) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_eight(void)
{
    if (ionic_is_power_of_two(8) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_fifteen(void)
{
    if (ionic_is_power_of_two(15) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_sixteen(void)
{
    if (ionic_is_power_of_two(16) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_1024(void)
{
    if (ionic_is_power_of_two(1024) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_1023(void)
{
    if (ionic_is_power_of_two(1023) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_is_power_of_two_large(void)
{
    if (ionic_is_power_of_two(1u << 31) != 1)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_down_zero(void)
{
    if (ionic_align_down_sz(0, 4) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_down_below_alignment(void)
{
    if (ionic_align_down_sz(1, 4) != 0)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_down_sz(3, 4) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_down_already_aligned(void)
{
    if (ionic_align_down_sz(4, 4) != 4)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_down_sz(128, 64) != 128)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_down_round_down(void)
{
    if (ionic_align_down_sz(5, 4) != 4)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_down_sz(100, 64) != 64)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_down_sz(255, 64) != 192)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_up_zero(void)
{
    if (ionic_align_up_sz(0, 4) != 0)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_up_round_up(void)
{
    if (ionic_align_up_sz(1, 4) != 4)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_up_sz(3, 4) != 4)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_up_sz(5, 4) != 8)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_up_sz(100, 64) != 128)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

static int test_align_up_already_aligned(void)
{
    if (ionic_align_up_sz(4, 4) != 4)
        return IONIC_RESULT_FAILURE;
    if (ionic_align_up_sz(128, 64) != 128)
        return IONIC_RESULT_FAILURE;
    return IONIC_RESULT_SUCCESS;
}

int main(void)
{
    if (test_is_power_of_two_zero() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_one() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_two() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_three() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_four() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_eight() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_fifteen() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_sixteen() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_1024() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_1023() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_is_power_of_two_large() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    if (test_align_down_zero() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_align_down_below_alignment() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_align_down_already_aligned() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_align_down_round_down() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    if (test_align_up_zero() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_align_up_round_up() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_align_up_already_aligned() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}
