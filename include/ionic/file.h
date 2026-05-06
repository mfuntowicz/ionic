//
// Created by mfuntowicz on 3/30/26.
//

#ifndef IONIC_FILE_H
#define IONIC_FILE_H

#if defined(__linux__) || defined(__APPLE__)
#include <ionic/platform/linux/file.h>
#else
#error "Unsupported platform"
#endif
#endif //IONIC_FILE_H