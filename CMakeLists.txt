cmake_minimum_required (VERSION 3.0)
project (libstegim)

# file variables
set(SOURCES_PATH "src")
set(HEADERS_PATH "include")
file(GLOB_RECURSE SOURCES "${SOURCES_PATH}/*.cpp")
file(GLOB_RECURSE HEADERS "${HEADERS_PATH}/*.cpp")

# find OpenCV
find_package(OpenCV REQUIRED)

# create stegim library
add_library (${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# include header directory
target_include_directories(${PROJECT_NAME}
	PUBLIC	${HEADERS_PATH}
	PRIVATE	${SOURCES_PATH})

# flags
target_compile_options(${PROJECT_NAME}
	PUBLIC -Wall -Wextra)

# to cmake induce the c++ standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)

# export information about compilation
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

# test
add_subdirectory(test)

enable_testing()
add_test (NAME lsb COMMAND lsb)
add_test (NAME lsbm COMMAND lsbm)
