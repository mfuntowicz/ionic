# Enable POSIX extensions for types like sigset_t
add_compile_definitions(_POSIX_C_SOURCE=200809L)

fetchcontent_declare(
        liburing
        GIT_REPOSITORY https://github.com/axboe/liburing
        GIT_TAG liburing-2.12
)
fetchcontent_getproperties(liburing)
if(NOT liburing_POPULATED)
    fetchcontent_makeavailable(liburing)

    set(LIBURING_CONFIGURE_CMD ./configure --cc=${CMAKE_C_COMPILER} --cxx=${CMAKE_CXX_COMPILER})
    if(IONIC_ENABLE_TESTS_SANITIZERS)
        list(APPEND LIBURING_CONFIGURE_CMD --enable-sanitizers)
    endif()
    
    add_custom_command(
        OUTPUT "${liburing_SOURCE_DIR}/Makefile"
        COMMAND sed -i "s/\\r//g" ./configure
        COMMAND chmod +x ./configure
        COMMAND ${LIBURING_CONFIGURE_CMD}
        # Pass CFLAGS/CXXFLAGS as env vars if needed, though liburing's configure might not grasp them all
        WORKING_DIRECTORY ${liburing_SOURCE_DIR}
        COMMENT "Configuring liburing"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${liburing_SOURCE_DIR}/src/liburing.a"
        COMMAND make -j library
        WORKING_DIRECTORY ${liburing_SOURCE_DIR}
        # This command depends on the Makefile existing (step 2)
        DEPENDS "${liburing_SOURCE_DIR}/Makefile"
        COMMENT "Building liburing.a"
        VERBATIM
    )

    add_custom_target(build_liburing_target
        DEPENDS "${liburing_SOURCE_DIR}/src/liburing.a"
    )

    add_library(liburing_static STATIC IMPORTED GLOBAL)
    set_target_properties(liburing_static PROPERTIES
        # Point to the SRC directory where the .a usually lands for liburing
        IMPORTED_LOCATION "${liburing_SOURCE_DIR}/src/liburing.a"
        INTERFACE_INCLUDE_DIRECTORIES "${liburing_SOURCE_DIR}/src/include"
    )

    add_dependencies(liburing_static build_liburing_target)
endif()