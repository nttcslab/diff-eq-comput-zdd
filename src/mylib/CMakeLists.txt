cmake_minimum_required(VERSION 3.8)
project(eqoptlibs)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(SRCS graph.cpp zdd.cpp)

add_library(eqoptlib STATIC ${SRCS})

find_library(ADEPTLIB NAMES adept REQUIRED)
target_link_libraries(eqoptlib ${ADEPTLIB})
target_compile_features(eqoptlib PUBLIC cxx_std_11)
