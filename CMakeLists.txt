cmake_minimum_required(VERSION 3.0)

project(parallel-hnsw)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)

add_executable(recallTable src/executables/recallTable.cpp)
target_compile_definitions(recallTable PRIVATE SRC_DIR="${CMAKE_SOURCE_DIR}/src")
target_include_directories(
	recallTable PUBLIC
	"${PROJECT_BINARY_DIR}"
	"${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(recallTable PUBLIC chmLib)
