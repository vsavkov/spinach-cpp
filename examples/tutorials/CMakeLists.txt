cmake_minimum_required(VERSION 3.30)
project(spinach_cpp_tutorials)

set(CMAKE_CXX_STANDARD 17)

#set(EIGEN_ROOT "path/to/eigen/library")
if("${EIGEN_ROOT}" STREQUAL "")
    message(FATAL_ERROR "EIGEN_ROOT is not defined")
else()
    message(STATUS "EIGEN_ROOT: ${EIGEN_ROOT}")
    set(EIGEN_INCLUDE_DIRS ${EIGEN_ROOT})
    message(STATUS "EIGEN_INCLUDE_DIRS: ${EIGEN_INCLUDE_DIRS}")
endif ()

find_package(ROOT)

add_executable(getting_started_with_sims getting_started_with_sims.cpp)
include_directories(getting_started_with_sims
        ${EIGEN_INCLUDE_DIRS}
        ${ROOT_INCLUDE_DIRS}
)
target_link_libraries(getting_started_with_sims PUBLIC
        ROOT::Core
        ROOT::Graf
        ROOT::Graf3d
)

message(STATUS "ROOT_LIBRARIES: ${ROOT_LIBRARIES}")
message(STATUS "ROOT_Core_LIBRARY: ${ROOT_Core_LIBRARY}")
