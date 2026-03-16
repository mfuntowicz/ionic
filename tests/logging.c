#include <stdlib.h>
#include <ionic/ionic.h>
#include "helpers.h"


int test_logging_error(void)
{
    setenv("IONIC_LOG", "error", 1);

    struct ionic_logger logger;
    ionic_logger_init(&logger);

    unsetenv("IONIC_LOG");

    if (logger.level != IONIC_LOG_LEVEL_ERROR)
        return IONIC_RESULT_FAILURE;

    if(!ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_ERROR))
        return IONIC_RESULT_FAILURE;

    if (ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_WARN) 
        || ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_INFO)
        || ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_DEBUG)
        || ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_TRACE))
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}

int test_logging_warn(void)
{
    setenv("IONIC_LOG", "warn", 1);

    struct ionic_logger logger;
    ionic_logger_init(&logger);

    unsetenv("IONIC_LOG");

    if (logger.level != IONIC_LOG_LEVEL_WARN)
        return IONIC_RESULT_FAILURE;

    if (ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_INFO)
        || ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_TRACE)
        || ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_DEBUG))
        return IONIC_RESULT_FAILURE;

    if (!ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_WARN) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_ERROR))
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}


int test_logging_info(void)
{
    setenv("IONIC_LOG", "info", 1);

    struct ionic_logger logger;
    ionic_logger_init(&logger);

    unsetenv("IONIC_LOG");

    if (logger.level != IONIC_LOG_LEVEL_INFO)
        return IONIC_RESULT_FAILURE;

    if (ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_TRACE)
        || ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_DEBUG))
        return IONIC_RESULT_FAILURE;

    if (!ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_INFO) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_WARN) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_ERROR))
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}

int test_logging_debug(void)
{
    setenv("IONIC_LOG", "debug", 1);

    struct ionic_logger logger;
    ionic_logger_init(&logger);

    unsetenv("IONIC_LOG");

    if (logger.level != IONIC_LOG_LEVEL_DEBUG)
        return IONIC_RESULT_FAILURE;

    if (ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_TRACE))
        return IONIC_RESULT_FAILURE;

    if (!ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_DEBUG)
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_INFO) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_WARN) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_ERROR))
        return IONIC_RESULT_FAILURE;


    return IONIC_RESULT_SUCCESS;
}

int test_logging_trace(void)
{
    setenv("IONIC_LOG", "trace", 1);

    struct ionic_logger logger;
    ionic_logger_init(&logger);

    unsetenv("IONIC_LOG");
    
    if (logger.level != IONIC_LOG_LEVEL_TRACE)
        return IONIC_RESULT_FAILURE;

    if (!ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_TRACE)
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_DEBUG) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_INFO) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_WARN) 
        || !ionic_log_level_is_enabled(&logger, IONIC_LOG_LEVEL_ERROR))
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}


int main(void) {
    if (test_logging_trace() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_logging_debug() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_logging_info() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_logging_warn() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;
    if (test_logging_error() != IONIC_RESULT_SUCCESS)
        return IONIC_RESULT_FAILURE;

    return IONIC_RESULT_SUCCESS;
}