fetchcontent_declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.12.0
)

fetchcontent_makeavailable(yyjson)

target_compile_definitions(yyjson PUBLIC -DYYJSON_STATIC)
target_compile_definitions(yyjson PUBLIC -DYYJSON_DISABLE_UTILS=ON)
target_compile_definitions(yyjson PUBLIC -DYYJSON_DISABLE_NON_STANDARD=ON)
target_compile_definitions(yyjson PUBLIC -DYYJSON_DISABLE_WRITER=ON)
target_compile_definitions(yyjson PUBLIC -DYYJSON_READER_DEPTH_LIMIT=5)

