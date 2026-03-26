#include <ionic/ionic.h>

void   ionic_iouring_init(struct ionic_context *);
void   ionic_iouring_destroy(struct ionic_context *);
size_t ionic_iouring_read(struct ionic_context *, int, unsigned char *, size_t, size_t);

/* Async I/O operations */
int    ionic_iouring_submit_read(struct ionic_context *, int fd, size_t offset, size_t len, void *userdata);
size_t ionic_iouring_poll_completions(struct ionic_context *, struct ionic_io_completion *completions, size_t max_completions);
size_t ionic_iouring_get_inflight(struct ionic_context *);
void  *ionic_iouring_acquire_staging(struct ionic_context *, size_t len, size_t *slot);
void   ionic_iouring_release_staging(struct ionic_context *, size_t slot);

#ifdef __IONIC_CUDA_ENABLED__
void ionic_iouring_cuda_pipeline_destroy(struct ionic_context *);
#endif