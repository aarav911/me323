> **Note.** These are the project's original design notes, written by Aarav Jayalwal at the start
> (tool choices, the planned pipeline). Parts of the "Current status" and "Planned pipeline" sections
> are out of date: the pipeline has since been built. For what the code does now and where
> everything is, see the [README](../README.md).

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

## 6. Stretch fit (optional)

One scale for all three axes cannot match a stock helmet to a particular
head: with the Run_2 scan the top gets no room for lattice at Gap 5 mm while
the low sides get 17 mm, and evening that out by raising Gap makes the helmet
278 × 301 mm. The stretch fit (`src/helmet_fit.cpp`, `src/fit_config.cpp`)
stretches the helmet along its own three axes, tilts it about the
side-to-side axis (+ = front up), and places its box centre at the head's
box centre plus an offset. The values are chosen by
`analysis_tools/stretch_fit_study.py` (outside this repo; it searches for the
most even room for lattice within agreed limits) and handed over in
`fit_config.txt`:

``` text
ME323_FIT_CONFIG 2
fit_mode stretch
head_scale 1
helmet_scale 1
gap 0.005
head_translation 0 0 0
helmet_translation 0 0 0
helmet_stretch <sx> <sy> <sz>
helmet_pitch <degrees>
helmet_offset <x> <y> <z>
```

Version 2 is only written in stretch mode; zoom mode still writes version 1,
byte for byte as before, so older builds keep reading it (and refuse version 2
rather than misreading it). `--fit zoom|stretch` overrides the file's mode, and
the viewer has a Zoom / Stretch switch with the stretch values editable.
`--lattice-room <size>` (model units, 0.02 = 20 mm) adds a target to the
"Room for lattice" line the generator prints for every fit.

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

The current prototype implements two implicit options in
`src/lattice_generator.cpp`:

-   **Auxetic:** a re-entrant honeycomb strut network repeated through the
    third axis.
-   **TPMS:** a gyroid sheet defined by its periodic implicit equation.

Both are clipped to the space between the fitted head and the inside of the
helmet shell. "Inside the helmet" comes from a radius map built in
`src/helmet_cavity.cpp`: rays from the head's centre record where they first
meet the shell's inner face; small holes in that map (vents) are bridged with a
smooth fill so lattice continues under them, and the one large hole (the
opening the head goes through) is left open. The shell is open, so a signed
distance to it has no meaningful inside; an earlier version used that sign plus
the helmet's bounding box and filled the whole box with lattice. The map needs
the shell to be star-shaped from the head centre, and generation stops with an
error if it is not. The input helmet surface is given a thin implicit skin, and a smooth
head-facing liner is added so clipped struts do not end as exposed rods
against the wearer. Both skins and the lattice are unioned into one sampled
solid.

The bottom stays open, so the helmet can be put on:

-   **A scanned head ends at a rim.** `sfm/export_for_me323.py` closes it with
    a lid (a fan of triangles) and records which triangles are lid in the
    STL's 80-byte header: `ME323-HEAD v1 lid-faces <first> <count>`
    (`src/head_lid.cpp` reads and checks it; a head without the marker, such
    as the GrabCAD model, is treated as complete). Directions from the head
    centre that see only the lid get no liner or lattice, just like the
    opening: beyond the rim lies head nobody measured. The liner used to wrap
    the whole head, lid included, which sealed the helmet shut.
-   **The liner is cut by direction only:** it exists where the helmet covers
    the head, vents included, however close the shell comes.
-   **The shell is the stock wall** (`--shell wall`, the default): the inside of
    the helmet mesh, found by winding number, plus a thin skin of
    `--shell-skin` grid cells (default 0.5) on every face so walls thinner than
    the grid can resolve do not open up. The lattice reaches the wall's inner
    face and fuses with the skin. The generator checks that the inside it finds
    adds up to the mesh's own volume and stops if not (a mesh that is not a
    closed solid): then use `--shell band`, the original, which pads every face
    by half a skin thickness. Band roughly doubles the wall's material and
    leaves hollows where the wall is thicker than the band; for the Run_2 stretch
    fit, Wall cuts the shell from 1116 to 890 cm³ and removes 54 cm³ of sealed
    pockets.
-   **It has to slide on.** `src/donning.cpp` tries straight lift-off
    directions (tilts up to 45 degrees forward/back, 10 sideways) and picks
    the one the head blocks least. Liner and lattice are then built against
    the space the head sweeps through along it, instead of the head itself,
    so the back of the skull, which curves under, cannot hook them. For the
    Run_2 scan, lifting straight up is blocked (about 6% of where material
    can go); about 20 degrees toward the back is clear. The shell is the
    stock helmet and is not trimmed; the program warns if it is in the way,
    or if it comes within the head clearance (a fit that is too tight).

Unit cells are phased from the combined model bounds (not world zero),
and the lattice pitch is reduced when the sampled gap is narrower. Marching
cubes extracts the result and the app writes `final_helmet_auxetic.stl` or
`final_helmet_tpms.stl` in the current working directory. The TPMS case is
generated when the app starts; either case can be regenerated from the
viewer. The generated STL is shown in the viewer.

Run `./build/me323` to generate the printable helmet and open it (the
defaults since step 4: the re-entrant honeycomb walls on a 0.37 mm grid,
simplified to 0.05 mm, about 11 minutes for our scan), or
`./build/me323 --lattice tpms` for the gyroid case. `--classic` gives back the
original behaviour in one flag (Aarav's cell, the coarse grid, no
simplification; flags after it still apply), e.g.
`./build/me323 --classic --lattice auxetic`. Add `--generate-only` to export the selected case and exit without
opening a viewer (useful for batch runs or automated checks). Add
`--export-fit` to also write `fitted_head.stl` and `fitted_helmet.stl`, the
exact meshes the generator received (the head keeps its lid marker).

`--fine-spacing <size>` (head-file units; 0.0005 = 0.5 mm) takes the surface
from a finer grid: the coarse grid (about head height / 120, capped at 4 M
samples) divided k times, k the smallest whole number giving at most the
requested spacing. Only the blocks the shell, liner or lattice surface can pass
through are sampled, in parallel; the fit, cavity, lift-off and cell pitch stay
on the coarse grid, and the wall skin is half a fine cell. Without the flag the
output is exactly what it was. For the Run_2 stretch fit, 0.0005 gives a
0.46 mm surface grid, 36.5 M triangles (a 1.83 GB STL) in about 90 s using up to
13 GB of memory. `--fine-exact`, `--fine-all-blocks` and `--fine-block-cells <n>`
are for testing (exact distances everywhere, no skipped blocks, block size); the
mesh must not depend on them.

`--cell reentrant` replaces the auxetic lattice with the textbook re-entrant
honeycomb (rows offset by half a repeat) drawn in the horizontal plane and
extruded straight up as continuous walls, which an FDM printer builds layer on
layer without overhangs: `--cell-width` (repeat side to side, default 0.010),
`--cell-angle` (default -35 degrees), `--cell-h-ratio` (h / l, default the value
giving an ideal Poisson's ratio of -1), `--wall-thickness` (default 0.0012),
`--line-width` (default 0.0004, only to warn when the wall is not a whole number
of extrusion lines). It needs `--fine-spacing` (the walls are thinner than the
coarse grid); the program refuses a cell whose walls would nearly touch, a grid
too coarse for the walls, and sizes that look like the wrong units. Check a cell
first with `analysis_tools/poisson_frame.py --cell-width 10 --cell-angle -35
--wall-thickness 1.2` (mm): Poisson's ratio both ways, stiffness, gap. It is the
default (`--cell original` gives the original lattice, unchanged); it applies to
the auxetic lattice only (the gyroid ignores it). The pattern is anchored on the
head's centre, so it is mirror-symmetric left to right.

`--roof-chamfer <deg>` (0 = none, default; else 45-80) puts slopes under the
liner from every wall, so that printed crown-down the liner rests on them
instead of bridging the cells (about 4.4 mm wide). Use it if the TPU test
coupon shows the roof sagging (below).

`--simplify <tolerance>` (e.g. 0.00005 = 0.05 mm) makes a printable mesh: each
block is simplified as soon as it is meshed, with a guaranteed bound (every
original point within the tolerance of the result and every new vertex within it
of the original), no self-intersections, the blocks still matching exactly, and
no two vertices close enough to merge in the 32-bit STL. For the Run_2 stretch fit,
`--fine-spacing 0.0004 --cell reentrant --simplify 0.00005` (the defaults) gives
5.9 M triangles (a 293 MB STL; unsimplified it would be about 119 M) in about 11
minutes with 2.2 GB of memory. Printable mode also removes loose fragments (keeps
the main body, its sealed air pockets, and any larger separate piece, with a
warning) and writes a second file for the printer, `final_helmet_<case>_print.stl`:
in millimetres, turned crown-down (a proper rotation, not a mirror), on the bed.

Printing (FDM, TPU): load `*_print.stl` as it is (crown on the bed, walls
vertical). Supports **on the build plate only**: the honeycomb cells are sealed,
and support inside them could never be removed; outside, the shell's overhangs
are reachable from the plate. Add a brim (the crown is a small contact). With a
0.4 mm nozzle the 1.2 mm walls are 3 lines.

Test coupon first: `./build/me323 --coupon` writes `coupon_A_no_chamfer.stl` and
`coupon_B_chamfer.stl` (40 x 40 x 32.7 mm, mm, ready to print; no head needed):
the helmet's walls between a floor and a liner-like roof, closed at the sides.
Print both in TPU. If A's roof is flat and solid, print the helmet as it is; if
it sags into the cells, regenerate the helmet with `--roof-chamfer 55` (B's roof
needs no bridging).

`tools/check_output.py` independently checks an output (Python with numpy,
scipy, open3d; matplotlib for `--png`), with Open3D rather than libigl. It sorts
the solid material into helmet skin, head-side liner, lattice inside the
helmet, and material outside the helmet. Step 1 fails if more than 0.1% is
outside the helmet. Step 2 fails if more than 0.1% is liner or lattice where
the head was not scanned, liner where the helmet does not cover, or material
the head passes through for the best straight lift-off direction. It also
checks the shell (step 3b): rays aimed at the head from outside must meet the
output near the stock outer face (no holes), and the middle of the stock wall
must be solid; and it estimates the mass (`--density`, default 1.0 g/cm³).
`--shell` / `--shell-skin` / `--fine-spacing` must match the generator's. The
checker also prints the mesh's exact enclosed volume, and with `--overhangs` the
area an FDM print would have to bridge or support printed crown-up or crown-down:

``` bash
./build/me323 --lattice auxetic --generate-only --export-fit
python tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl --png slices.png
```

`--step3 --zone-helmet data/helmet.STL` adds the step-3 report: room for
lattice measured along the head's normal over a fixed protection zone (what
the default Gap-5 zoom fit covers), and a gate (room ≥ 20 mm, p95/p5 ≤ 1.5,
size ≤ 245 × 280 mm, zone covered ≥ 99%, front edge not lower, shell not in the
way). The gate counts in the exit status.

This is a first printable-mesh prototype, not a validated helmet design.
The union is voxel-resampled, so surfaces and thin features are approximate;
the source meshes must enclose meaningful volumes for signed-distance
inside/outside classification. Inspect the exported STL in a slicer and
validate its wall thickness, clearances, manifoldness, and mechanical
performance before considering manufacture or wear.

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
