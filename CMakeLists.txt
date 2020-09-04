cmake_minimum_required(VERSION 3.12)

project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    find_package(Python 3.6 COMPONENTS Development REQUIRED)
else()
    find_package(Python 3.7 COMPONENTS Development REQUIRED)
endif()
add_subdirectory(pybind11)

option(DIFFVG_CUDA "Build diffvg with GPU code path?" ON)

if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support")
    find_package(CUDA 10 REQUIRED)
    set(CMAKE_CUDA_STANDARD 11)
    if(NOT WIN32)
        # Hack: for some reason the line above doesn't work on some Linux systems.
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
        #set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
    endif()
else()
    message(STATUS "Build without CUDA support")
    find_package(Thrust REQUIRED)
endif()

# include_directories(${CMAKE_SOURCE_DIR}/pybind11/include)
include_directories(${PYTHON_INCLUDE_PATH})
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(pybind11/include)
if(DIFFVG_CUDA)
    link_directories(${CUDA_LIBRARIES})
else()
    include_directories(${THRUST_INCLUDE_DIR})
endif()

if(NOT MSVC)
  # These compile definitions are not meaningful for MSVC
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
endif()

if(NOT DIFFVG_CUDA)
    add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS atomic.h
         color.h
         cdf.h
         cuda_utils.h
         diffvg.h
         edge_query.h
         filter.h
         matrix.h
         parallel.h
         pcg.h
         ptr.h
         sample_boundary.h
         scene.h
         shape.h
         solve.h
         vector.h
         within_distance.h
         winding_number.h
         atomic.cpp
         color.cpp
         diffvg.cpp
         parallel.cpp
         scene.cpp
         shape.cpp)

if(DIFFVG_CUDA)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        diffvg.cpp
        scene.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    cuda_add_library(diffvg MODULE ${SRCS})
else()
    add_library(diffvg MODULE ${SRCS})
endif()

if(APPLE)
    # The "-undefined dynamic_lookup" is a hack for systems with
    # multiple Python installed. If we link a particular Python version
    # here, and we import it with a different Python version later.
    # likely a segmentation fault.
    # The solution for Linux Mac OS machines, as mentioned in 
    # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
    # is to not link against Python library at all and resolve the symbols
    # at compile time.
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()

target_link_libraries(diffvg ${DYNAMIC_LOOKUP})

if(WIN32)
    # See: https://pybind11.readthedocs.io/en/master/compiling.html#advanced-interface-library-target
    target_link_libraries(diffvg pybind11::module)
    set_target_properties(diffvg PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

set_target_properties(diffvg PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(diffvg PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(UNIX AND NOT APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "@loader_path")
endif()

set_property(TARGET diffvg PROPERTY CXX_STANDARD 11)
set_target_properties(diffvg PROPERTIES PREFIX "")
# Still enable assertion in release mode
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

if(NOT WIN32)
    find_package(TensorFlow)
    if(TensorFlow_FOUND)
        add_subdirectory(pydiffvg_tensorflow/custom_ops)
    else()
        message(INFO " Building without TensorFlow support (not found)")
    endif()
endif()
