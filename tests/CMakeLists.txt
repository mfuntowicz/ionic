add_executable(test_version version.c)
target_link_libraries(test_version PRIVATE ionic)
target_include_directories(test_version PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_version COMMAND test_version)

add_executable(test_logging logging.c)
target_link_libraries(test_logging PRIVATE ionic)
target_include_directories(test_logging PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_logging COMMAND test_logging)

add_executable(test_utils utils.c)
target_link_libraries(test_utils PRIVATE ionic)
target_include_directories(test_utils PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_utils COMMAND test_utils)

add_executable(test_planner planner.c)
target_link_libraries(test_planner PRIVATE ionic)
target_include_directories(test_planner PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_planner COMMAND test_planner)
