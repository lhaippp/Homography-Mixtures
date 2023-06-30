cmake_minimum_required(VERSION 3.14)

project(SGridSearch)

message(STATUS "start running cmake...")

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(SGriSearch main.cpp SGridTracker.h SGridTracker.cpp)

target_link_libraries(SGriSearch ${Boost_LIBRARIES})

target_link_libraries(SGriSearch ${OpenCV_LIBS})