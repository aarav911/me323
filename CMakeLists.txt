cmake_minimum_required(VERSION 3.20)

project(me323 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.6.4
)

# We only need the viewer for now.
set(LIBIGL_OPENGL ON CACHE BOOL "" FORCE)
set(LIBIGL_GLFW ON CACHE BOOL "" FORCE)
set(LIBIGL_IMGUI ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(libigl)

add_executable(me323
    src/main.cpp
    src/lattice_generator.cpp
    src/helmet_cavity.cpp
    src/head_lid.cpp
    src/donning.cpp
    src/fit_config.cpp
    src/helmet_fit.cpp
    src/lattice_struts.cpp
    src/mesh_distance.cpp
    src/fine_grid.cpp
)

target_link_libraries(me323
    PRIVATE
    igl::core
    igl::glfw
    igl::imgui
)