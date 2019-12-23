cmake_minimum_required(VERSION 3.15)
project(datasets)

set(CMAKE_CXX_STANDARD 14)

find_package(HighFive 2.1 REQUIRED)
add_definitions(-DDEBUG)
add_executable(audio Audio.cpp)
target_link_libraries(audio HighFive xxhash)