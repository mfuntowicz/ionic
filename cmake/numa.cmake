if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(NUMA_INCLUDE_DIR NAMES numa.h)
    find_library(NUMA_LIBRARY NAMES numa)
    if(NOT NUMA_INCLUDE_DIR OR NOT NUMA_LIBRARY)
        message(FATAL_ERROR "libnuma not found. Install numactl-devel (Fedora/RHEL) or libnuma-dev (Debian/Ubuntu)")
    endif()
    add_library(NUMA::numa SHARED IMPORTED GLOBAL)
    set_target_properties(NUMA::numa PROPERTIES
        IMPORTED_LOCATION "${NUMA_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
    )
    list(APPEND IONIC_LIBS NUMA::numa)
endif()