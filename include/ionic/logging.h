#ifndef IONIC_LOGGING_H
#define IONIC_LOGGING_H

#include <stdarg.h>

enum ionic_log_level {
    IONIC_LOG_LEVEL_OFF = 0,
    IONIC_LOG_LEVEL_ERROR = 1,
    IONIC_LOG_LEVEL_WARN = 2,
    IONIC_LOG_LEVEL_INFO = 3,
    IONIC_LOG_LEVEL_DEBUG = 4,
    IONIC_LOG_LEVEL_TRACE = 5,
};
typedef enum ionic_log_level ionic_log_level_t;

struct ionic_logger {
    ionic_log_level_t level;
    unsigned char ready;
};
typedef struct ionic_logger ionic_logger_t;


#define IONIC_LOG(logger, lvl, tag, ...) ionic_log((logger), (lvl), (tag), __VA_ARGS__)
#define IONIC_TRACE(logger, tag, ...) IONIC_LOG((logger), IONIC_LOG_LEVEL_TRACE, (tag), __VA_ARGS__)
#define IONIC_DEBUG(logger, tag, ...) IONIC_LOG((logger), IONIC_LOG_LEVEL_DEBUG, (tag), __VA_ARGS__)
#define IONIC_INFO(logger, tag, ...)  IONIC_LOG((logger), IONIC_LOG_LEVEL_INFO,  (tag), __VA_ARGS__)
#define IONIC_WARN(logger, tag, ...)  IONIC_LOG((logger), IONIC_LOG_LEVEL_WARN,  (tag), __VA_ARGS__)
#define IONIC_ERROR(logger, tag, ...) IONIC_LOG((logger), IONIC_LOG_LEVEL_ERROR, (tag), __VA_ARGS__)

#endif // IONIC_LOGGING_H
