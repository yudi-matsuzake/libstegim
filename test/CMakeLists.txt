cmake_minimum_required(VERSION 3.0)
project(libstegim-test)

add_definitions("-DCOVER_IMAGE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/cover\"")

# lsb
add_executable(lsb lsb.cpp)
target_link_libraries(lsb libstegim)

# flags
target_compile_options(lsb
	PUBLIC -Wall -Wextra)

