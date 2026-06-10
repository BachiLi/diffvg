cmake_minimum_required(VERSION 3.18)

project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics" LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
    find_package(Python 3.6 COMPONENTS Interpreter Development.Module REQUIRED)
else()
    find_package(Python 3.7 COMPONENTS Interpreter Development.Module REQUIRED)
endif() 
#add_subdirectory(pybind11)
find_package(pybind11 CONFIG REQUIRED)
find_package(Protobuf REQUIRED)

# Check if compiling on macOS to disable CUDA by default
if(APPLE)
    option(DIFFVG_CUDA "Build diffvg with GPU code path?" OFF)
else()
    option(DIFFVG_CUDA "Build diffvg with GPU code path?" ON)
endif()

#option(DIFFVG_CUDA "Build diffvg with GPU code path?" ON)

if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support")
    enable_language(CUDA)
    find_package(CUDAToolkit 10 REQUIRED)
else()
    message(STATUS "Build without CUDA support")
    find_package(Thrust REQUIRED)
endif()

set(SRCS
    #atomic.h
    #color.h
    #cdf.h
    #cuda_utils.h
    #diffvg.h
    #edge_query.h
    #filter.h
    #matrix.h
    #parallel.h
    #pcg.h
    #ptr.h
    #sample_boundary.h
    #scene.h
    #shape.h
    #solve.h
    #vector.h
    #within_distance.h
    #winding_number.h
    atomic.cpp
    color.cpp
    diffvg.cpp
    parallel.cpp
    scene.cpp
    shape.cpp
    bindings.cpp
)

add_library(diffvg MODULE ${SRCS})

target_include_directories(diffvg PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
)

  
if(DIFFVG_CUDA)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_compile_definitions(diffvg PRIVATE COMPILE_WITH_CUDA)

    set_source_files_properties(
        diffvg.cpp
        scene.cpp
        PROPERTIES LANGUAGE CUDA
    )

    set_target_properties(diffvg PROPERTIES
      CUDA_ARCHITECTURES native
    )

    target_link_libraries(diffvg PRIVATE CUDA::cudart)
else()
    target_include_directories(diffvg PRIVATE ${THRUST_INCLUDE_DIR})
    target_compile_definitions(diffvg PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
endif()

if(NOT MSVC)
    target_compile_options(diffvg PRIVATE -Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
    target_compile_options(diffvg PRIVATE /Wall /Zi)
    target_link_options(diffvg PRIVATE /DEBUG)
endif()

if(APPLE)
    target_link_options(diffvg PRIVATE -undefined dynamic_lookup)
endif()

target_link_libraries(diffvg PRIVATE pybind11::module)

if(WIN32)
    set_target_properties(diffvg PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()

set_target_properties(diffvg PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    PREFIX ""
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

if(UNIX AND NOT APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "@loader_path")
endif()

# Still enable assertion in release mode
string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
