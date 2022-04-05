file(GLOB headers chm/*.hpp)
file(GLOB sources chm/*.cpp)
source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})

add_library(chmLib ${headers} ${sources})
target_include_directories(chmLib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
