cmake_minimum_required(VERSION 3.26)
project(ionic VERSION 0.1.0 LANGUAGES C)

include(FetchContent)

option(IONIC_BUILD_EXAMPLES "Build examples" OFF)
option(IONIC_BUILD_TESTS "Build tests" OFF)
option(IONIC_ENABLE_TESTS_SANITIZERS "Flag indicating if cmake should enable sanitizers" OFF)

option(IONIC_ENABLE_CUDA "Enable CUDA support" OFF)
option(IONIC_ENABLE_SAFETENSORS "Enable Safetensors support" OFF)

set(IONIC_LIBS "")
set(IONIC_SOURCES lib/context.c lib/memory.c lib/logging.c)
set(IONIC_HEADERS include/ionic/ionic.h include/ionic/types.h include/ionic/logging.h)

if(IONIC_ENABLE_CUDA)
    message(STATUS "CUDA support enabled")
    find_package(CUDAToolkit REQUIRED COMPONENTS CUDA::cudart)
    list(APPEND IONIC_LIBS CUDA::cudart)
endif()

if(IONIC_ENABLE_SAFETENSORS)
    message(STATUS "Safetensors support enabled")
    
    include(cmake/yyjson.cmake)
    list(APPEND IONIC_LIBS yyjson)
    list(APPEND IONIC_SOURCES lib/safetensors.c)

    add_compile_definitions(__IONIC_SAFETENSORS_ENABLED__=1)
    add_compile_definitions(__IONIC_TENSORS_ENABLED__=1)
endif()

add_library(ionic STATIC ${IONIC_SOURCES} ${IONIC_HEADERS})
target_link_libraries(ionic PRIVATE ${IONIC_LIBS})
target_include_directories(ionic PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(IONIC_BUILD_EXAMPLES)
    add_subdirectory(examples)
    message(STATUS "Examples enabled")
endif()

if(IONIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)

    message(STATUS "Tests enabled")

    # Enable sanitizers for tests
    if($IONIC_ENABLE_TESTS_SANITIZERS AND NOT MSVC)
        target_compile_options(ionic_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(ionic_tests PRIVATE -fsanitize=address,undefined)
    endif()
endif()