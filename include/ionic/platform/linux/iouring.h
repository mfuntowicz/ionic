#include <ionic/ionic.h>

void   ionic_iouring_init(struct ionic_context *);
void   ionic_iouring_destroy(struct ionic_context *);
size_t ionic_iouring_read(struct ionic_context *, int, unsigned char *, size_t, size_t);

#ifdef __IONIC_CUDA_ENABLED__
void ionic_iouring_cuda_pipeline_destroy(struct ionic_context *);
#endif