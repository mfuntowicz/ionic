#include "ionic/ionic.h"
#include "ionic/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static const char *const LEVEL_NAMES[] = {
    "OFF", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"
};

static inline ionic_log_level_t ionic_log_level_from_str(const char *s) {
    if (!s) return IONIC_LOG_LEVEL_OFF;

    if (strcmp(s, "trace") == 0 || strcmp(s, "TRACE") == 0) return IONIC_LOG_LEVEL_TRACE;
    if (strcmp(s, "debug") == 0 || strcmp(s, "DEBUG") == 0) return IONIC_LOG_LEVEL_DEBUG;
    if (strcmp(s, "info")  == 0 || strcmp(s, "INFO")  == 0) return IONIC_LOG_LEVEL_INFO;
    if (strcmp(s, "warn")  == 0 || strcmp(s, "WARN")  == 0) return IONIC_LOG_LEVEL_WARN;
    if (strcmp(s, "error") == 0 || strcmp(s, "ERROR") == 0) return IONIC_LOG_LEVEL_ERROR;
    if (strcmp(s, "off")   == 0 || strcmp(s, "OFF")   == 0) return IONIC_LOG_LEVEL_OFF;

    return IONIC_LOG_LEVEL_OFF;
}

unsigned char ionic_log_level_is_enabled(ionic_logger_t *logger, ionic_log_level_t level) {
    printf("logger->level: %d, level: %d\n", logger->level, level);
    return level <= logger->level;
}

void ionic_logger_init(ionic_logger_t *logger) {
    const char *env = getenv("IONIC_LOG");
    logger->level = ionic_log_level_from_str(env);
    logger->ready = 1;
}

void ionic_log(ionic_logger_t *logger, ionic_log_level_t level, const char *tag, const char *fmt, ...) {
    if (!logger->ready) ionic_logger_init(logger);
    if (level > logger->level) return;

    fprintf(stderr, "[%-6s] [%-15s] ", LEVEL_NAMES[level], tag);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fputc('\n', stderr);
}
