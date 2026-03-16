#include <ionic/ionic.h>
#include "helpers.h"

int main(void) {
    if (ionic_version() != IONIC_VERSION)
        return IONIC_RESULT_FAILURE;
    
    return IONIC_RESULT_SUCCESS;
}