add_executable(safetensors safetensors.c)
target_link_libraries(safetensors PRIVATE ionic)
target_include_directories(safetensors PRIVATE ${CMAKE_SOURCE_DIR}/include)
