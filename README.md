# ME323 --- Helmet Lattice / Metamaterial Generator

## Project Overview

This project is a C++ computational-geometry pipeline for designing a
**3D lattice/metamaterial structure between a human head and a helmet**.

The intended workflow is:

``` text
Head STL ───────┐
                │
                ▼
          Geometry processing
                │
                ▼
       Define the gap / volume
                │
                ▼
        Generate lattice structure
                │
                ▼
       Combine / trim / validate
                │
                ▼
          Printable STL
                ▲
                │
Helmet STL ─────┘
```

The immediate goal is to establish a reliable geometry-processing
environment and correctly load, inspect, scale, and align the input STL
meshes. Lattice generation and manufacturing/export operations will be
built on top of this foundation.

------------------------------------------------------------------------

## Core Problem

Given:

-   a **head surface mesh**
-   a **helmet surface mesh**

we want to determine the usable 3D region between them.

That region will eventually become the domain in which a
lattice/metamaterial structure is generated.

Conceptually:

``` text
             OUTER HELMET
        ┌─────────────────────┐
       /                       \
      /     LATTICE REGION      \
     |      ┌─────────────┐     |
     |      │    HEAD     │     |
     |      └─────────────┘     |
      \                         /
       └───────────────────────┘
```

The important distinction is that the project is **not simply a mesh
viewer**. The viewer is an early development tool that lets us verify
the geometry before performing computationally expensive operations.

------------------------------------------------------------------------

# Technology Stack

## C++20

The core application is written in **C++20**.

### Why C++?

The eventual workload involves:

-   large triangle meshes
-   spatial queries
-   collision/intersection tests
-   distance calculations
-   mesh generation
-   geometric transformations
-   potentially millions of lattice elements

C++ gives us:

-   high performance
-   direct memory control
-   mature computational-geometry libraries
-   strong numerical libraries
-   a large ecosystem for mesh processing

C++ is therefore being used as the primary implementation language
rather than Python.

------------------------------------------------------------------------

# CMake

The project uses **CMake** as its build system.

Example build:

``` bash
cmake -S . -B build -G Ninja
cmake --build build
```

### Why CMake?

The project depends on several C++ libraries. CMake gives us a standard
way to:

-   locate installed libraries
-   configure compiler settings
-   manage targets
-   link dependencies
-   build the project consistently

It also makes the project easier to move to another machine later.

------------------------------------------------------------------------

# Ninja

**Ninja** is used as the CMake generator.

``` bash
-G Ninja
```

### Why Ninja?

Ninja is a fast build system designed for incremental compilation.

This becomes useful as the project grows from a single source file into
multiple geometry-processing modules.

For example:

``` text
src/
├── main.cpp
├── geometry.cpp
├── alignment.cpp
├── lattice.cpp
├── mesh_utils.cpp
└── export.cpp
```

After changing one file, Ninja can rebuild only what is necessary.

------------------------------------------------------------------------

# Eigen

The project uses **Eigen** for linear algebra.

Current installed version:

``` text
Eigen 3.4.0
```

Eigen provides data structures such as:

``` cpp
Eigen::Vector3d
Eigen::MatrixXd
Eigen::MatrixXi
```

These are particularly useful for representing mesh data.

A triangular mesh is commonly represented as:

``` text
V = vertex coordinates
F = triangle connectivity
```

For example:

``` cpp
Eigen::MatrixXd V;
Eigen::MatrixXi F;
```

where:

-   `V` contains the 3D coordinates of vertices
-   `F` specifies which vertices form each triangle

### Why Eigen?

The project will require substantial numerical computation:

-   vectors
-   matrices
-   transformations
-   rotations
-   coordinate systems
-   bounding boxes
-   distances
-   transformations of large sets of points

Eigen provides these operations efficiently and integrates naturally
with libigl.

------------------------------------------------------------------------

# CGAL

The project uses **CGAL --- Computational Geometry Algorithms Library**.

Current installed version:

``` text
CGAL 6.1.1
```

CGAL is intended to provide the project's more advanced
computational-geometry capabilities.

Potential uses include:

-   spatial data structures
-   geometric predicates
-   intersections
-   point/mesh queries
-   distance calculations
-   mesh operations
-   robust geometric algorithms

### Why CGAL?

The central difficulty of this project is ultimately geometric rather
than graphical.

We need to reason about questions such as:

> Is this point inside the helmet/head region?

> What is the distance from this point to a surface?

> Where do these surfaces intersect?

> What volume exists between two surfaces?

> Does this lattice element remain inside the desired domain?

These operations benefit from a mature computational-geometry library
rather than implementing everything from scratch.

CGAL therefore forms an important part of the computational foundation.

------------------------------------------------------------------------

# libigl

The project uses **libigl** for mesh processing and visualization.

The current project obtains libigl through CMake `FetchContent`.

The project currently uses:

``` text
libigl
├── igl::core
├── igl::opengl
└── igl::glfw
```

### Why libigl?

libigl is particularly convenient for working with triangle meshes
represented as:

``` text
V + F
```

It provides a large collection of mesh-processing algorithms and
utilities without requiring a heavyweight mesh class abstraction.

For this project, libigl is useful for:

-   loading STL meshes
-   representing meshes
-   mesh processing
-   visualization
-   debugging geometry

It also integrates naturally with Eigen.

------------------------------------------------------------------------

# GLFW / OpenGL

libigl's interactive viewer uses **GLFW** and **OpenGL**.

The current application can launch an interactive 3D viewer from C++:

``` cpp
igl::opengl::glfw::Viewer viewer;

viewer.launch();
```

### Why have a viewer?

The viewer is primarily a **development and debugging tool**.

Before performing complicated geometry operations, we need to be able to
visually inspect:

-   mesh orientation
-   scale
-   positioning
-   alignment
-   surface quality
-   intersections
-   generated lattice structures

A numerical algorithm can technically succeed while operating on
completely misaligned geometry.

Being able to immediately see the meshes is therefore extremely
valuable.

------------------------------------------------------------------------

# STL

STL is being used as the primary **input/output interchange format**.

Current input files:

``` text
data/
├── head.stl
└── helmet.stl
```

STL is convenient because it is widely supported by:

-   CAD software
-   3D scanners
-   mesh-processing software
-   3D-printing software
-   slicing software

However, STL is **not intended to be the internal representation** of
the geometry.

Internally, meshes are converted into:

``` text
V — vertex coordinates
F — triangle connectivity
```

This makes numerical and geometric operations much easier.

------------------------------------------------------------------------

# Current Project Structure

The project currently follows this structure:

``` text
me323/
│
├── CMakeLists.txt
│
├── data/
│   ├── head.stl
│   └── helmet.stl
│
├── src/
│   └── main.cpp
│
└── build/
```

The `build/` directory contains generated CMake/Ninja build files and
compiled binaries and should generally not be treated as source code.

As the project grows, the structure is expected to evolve toward
something like:

``` text
me323/
│
├── CMakeLists.txt
│
├── README.md
│
├── data/
│   ├── head.stl
│   └── helmet.stl
│
├── include/
│   ├── geometry/
│   ├── lattice/
│   └── visualization/
│
├── src/
│   ├── main.cpp
│   ├── geometry/
│   ├── lattice/
│   └── visualization/
│
├── tests/
│
└── build/
```

------------------------------------------------------------------------

# Current Geometry Pipeline

The current prototype performs the following operations:

## 1. Load the head mesh

The STL is loaded into:

``` cpp
Eigen::MatrixXd V_head;
Eigen::MatrixXi F_head;
```

## 2. Load the helmet mesh

Similarly:

``` cpp
Eigen::MatrixXd V_helmet;
Eigen::MatrixXi F_helmet;
```

## 3. Inspect bounding boxes

For each mesh we calculate:

``` text
minimum XYZ
maximum XYZ
size
center
```

This is important because STL files do not necessarily use the same
units or coordinate system.

For example, the current meshes were found to have approximately:

``` text
HEAD
Size:
0.255 × 0.438 × 0.265
```

while the original helmet mesh was approximately:

``` text
HELMET
Size:
126.8 × 111.7 × 136.9
```

This revealed a large unit/scale mismatch.

## 4. Scale the helmet

Rather than permanently hard-coding an arbitrary scale factor, the
current approach derives the scale from the geometry.

The basic idea is:

``` text
desired helmet height
────────────────────── = scale
original helmet height
```

This makes the alignment process reproducible and tunable.

## 5. Align the meshes

The current prototype uses bounding-box information to align:

-   X center
-   Z center
-   vertical position

and exposes additional X/Y/Z offsets for manual adjustment.

This is an initial registration method, not the final
geometric-registration algorithm.

------------------------------------------------------------------------

# Why Alignment Is Its Own Problem

Correct mesh alignment is critical.

A lattice generated between two incorrectly registered surfaces is
useless, even if the lattice-generation algorithm itself is
mathematically correct.

There are several possible sources of mismatch:

``` text
1. Different units
2. Different scale
3. Different origin
4. Different translation
5. Different orientation
6. Different coordinate conventions
```

The current prototype handles scale and translation first.

Rotation and more sophisticated surface-based registration can be added
if the input meshes require it.

------------------------------------------------------------------------

# Planned Geometry Pipeline

Once the head and helmet are correctly registered, the project can move
toward the actual objective.

## Stage 1 --- Mesh validation

Verify:

-   watertightness where required
-   triangle orientation
-   degenerate triangles
-   self-intersections
-   coordinate system
-   units
-   mesh resolution

------------------------------------------------------------------------

## Stage 2 --- Define the gap region

The goal is to determine the 3D region satisfying approximately:

``` text
outside head
AND
inside helmet
```

with appropriate clearance constraints.

Conceptually:

``` text
Helmet volume
      ∩
Outside(head)
      =
Lattice domain
```

This is the central geometric problem.

------------------------------------------------------------------------

## Stage 3 --- Generate the lattice

A lattice structure will be generated inside the available domain.

Possible lattice families can later include:

-   cubic
-   octet
-   tetrahedral
-   gyroid
-   TPMS-based structures
-   custom unit cells

The lattice should eventually be parameterized by quantities such as:

``` text
cell size
strut thickness
relative density
orientation
local stiffness
```

------------------------------------------------------------------------

## Stage 4 --- Clip lattice to the available domain

A generated lattice cannot simply occupy the entire bounding box.

It must be constrained by the actual head/helmet geometry.

The desired result is approximately:

``` text
          helmet
       ┌───────────┐
      /             \
     /  lattice      \
    |   ╔═══════╗     |
    |   ║  head ║     |
    |   ╚═══════╝     |
     \               /
      └─────────────┘
```

------------------------------------------------------------------------

## Stage 5 --- Mesh processing

The generated structure will need to be converted into a valid printable
mesh.

Potential operations include:

-   trimming
-   boolean operations
-   merging
-   remeshing
-   removing degenerate geometry
-   checking manifoldness
-   checking for self-intersections

------------------------------------------------------------------------

## Stage 6 --- Export

The final result should be exportable as an STL suitable for downstream
CAD/3D-printing workflows.

------------------------------------------------------------------------

# Design Philosophy

A few architectural choices are deliberate.

## Separate visualization from geometry

The viewer is useful, but it should not become the core of the project.

The computational pipeline should eventually be usable independently of
the GUI:

``` text
geometry
   ↓
processing
   ↓
lattice generation
   ↓
mesh generation
   ↓
STL export
```

The viewer can then visualize intermediate results.

------------------------------------------------------------------------

## Keep transformations explicit

Scale, rotation, and translation should be represented explicitly rather
than hidden inside mesh-loading code.

This makes it possible to reproduce a geometry-processing pipeline and
debug it.

Eventually, the transformation should be represented mathematically as
something like:

``` text
p' = R p + t
```

where:

-   `R` = rotation
-   `t` = translation

and scaling can be represented separately.

------------------------------------------------------------------------

## Use libraries for difficult primitives

The project should not reimplement mature algorithms unnecessarily.

The intended division of responsibility is approximately:

``` text
Eigen
    ↓
Linear algebra

CGAL
    ↓
Robust computational geometry

libigl
    ↓
Mesh processing + mesh utilities

OpenGL / GLFW
    ↓
Visualization

C++
    ↓
Project-specific algorithms
```

The project's own code should focus primarily on the **helmet/lattice
problem**, rather than rebuilding generic geometry infrastructure.

------------------------------------------------------------------------

# Current Status

The development environment is operational.

Currently working:

-   C++20 compilation
-   CMake
-   Ninja
-   Eigen
-   CGAL
-   libigl
-   GLFW/OpenGL viewer
-   WSL graphical output
-   STL loading
-   simultaneous visualization of head and helmet meshes
-   bounding-box analysis
-   automatic initial scaling/alignment

The current prototype is therefore at the **geometry-ingestion and
registration stage**.

The next major engineering problem is:

> **Precisely define and compute the 3D volume between the head and
> helmet surfaces.**

That volume will become the domain for the lattice/metamaterial
generator.

------------------------------------------------------------------------

# Build and Run

From the project root:

``` bash
cmake -S . -B build -G Ninja
cmake --build build
./build/me323
```

For a clean rebuild:

``` bash
rm -rf build
cmake -S . -B build -G Ninja
cmake --build build
./build/me323
```

------------------------------------------------------------------------

# Long-Term Goal

The eventual program should be able to accept something conceptually
like:

``` bash
./build/me323 data/head.stl data/helmet.stl
```

and perform a pipeline resembling:

``` text
          head.stl
              │
              ▼
       ┌──────────────┐
       │ Mesh loading │
       └──────┬───────┘
              │
              │
          helmet.stl
              │
              ▼
       ┌──────────────┐
       │ Registration │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Gap/domain   │
       │ computation  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   Lattice    │
       │  generation  │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ Mesh cleanup │
       │ & validation │
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │   STL export │
       └──────────────┘
```

The final objective is a **parameterized, computationally generated
helmet metamaterial/lattice structure**, rather than a manually modeled
mesh.
