# ME323 auxetic helmet generator

ME323 course project, IIT Bombay. This program takes a **3D scan of one person's head** and a
**stock helmet shell** (a bike helmet from GrabCAD), and fills the gap between them with an
**auxetic re-entrant honeycomb**: a lattice with a negative Poisson's ratio, which pulls material
in toward an impact instead of pushing it away. The output is one closed, printable STL.

| | |
|---|---|
| Lattice | re-entrant honeycomb walls: θ = −35°, 10 × 14.28 mm repeat, 1.2 mm walls; modelled ν ≈ −0.87 / −0.80 (the original cell: −0.02) |
| Padding | 21.5–38 mm between head and shell (median 30 mm) |
| Mesh | 5.65 M triangles, closed, no self-intersections, within 0.05 mm of the exact surface |
| Run time | about 11 minutes, 2.3 GB of memory (Apple M4 Pro) |

Branch `vt/main` holds the working pipeline. The first three commits are Aarav Jayalwal's original
generator; the rest build on it (see [History](#history)).

---

## Contents

1. [How it works, in one page](#how-it-works-in-one-page)
2. [Repository map](#repository-map)
3. [Source files](#source-files)
4. [How a run flows through the code](#how-a-run-flows-through-the-code)
5. [Build](#build)
6. [Run](#run)
7. [Check an output](#check-an-output)
8. [What is not in this repository](#what-is-not-in-this-repository)
9. [History](#history)
10. [Credits](#credits)

---

## How it works, in one page

Every part of the helmet is a **signed distance field**: a number at every point of space,
negative inside the part, zero on its surface, positive outside. Parts are combined with
`min` (union: inside either) and `max` (intersection: inside both), so the whole helmet is one
field:

```
helmet = min( shell, liner, max( walls, inside_shell, clear_of_head ) )
```

i.e. the shell, plus the liner, plus the honeycomb walls *only where* they are inside the shell
*and* clear of the head's path. The helmet's surface is wherever this field is zero.

1. **Fit the shell.** The stock helmet is stretched along its own axes, tilted and shifted
   so there is at least 20 mm of room for padding everywhere (`helmet_fit`, `fit_config`).
2. **Inside the shell.** Looking out from the head's centre, the distance to the shell's inner
   face is recorded for every direction (a radius map); vents are bridged by a smooth fill,
   the bottom opening is left open (`helmet_cavity`).
3. **Only where the head was scanned.** The scan is closed underneath by a "lid" that is not
   real head; directions that only see the lid get no padding (`head_lid`).
4. **It can be put on.** The program tries 95 straight lift-off directions, picks the one the
   head can slide out along, and keeps the head's whole swept path empty (`donning`).
5. **The honeycomb.** Re-entrant walls drawn in the horizontal plane and extruded straight up,
   rows offset by half a cell so the slanted walls fold like hinges (`lattice_struts`).
6. **Surface and mesh.** The field is sampled on a 0.37 mm grid, only in blocks a surface can
   pass through, turned into triangles by marching tetrahedra, and simplified to within
   0.05 mm (`fine_grid`, `mesh_distance`, `mesh_simplify`).
7. **Print file.** Loose fragments are removed, and a print copy is written in millimetres,
   turned crown-down and set on the bed (`mesh_components`, `print_export`).
8. **Independent check.** A separate Python checker, built on a different library, verifies
   nothing is outside the helmet, the head can lift off, the shell is whole and the fit passes
   its gate (`tools/check_output.py`).

---

## Repository map

```
.
├── README.md                 this file
├── CMakeLists.txt            build: C++20, libigl v2.6.4 (with Eigen, GLFW, ImGui) fetched automatically
├── src/                      the generator (C++), see "Source files" below
│   ├── main.cpp              entry point: command line, fit, viewer, writes the output files
│   ├── fit_config.*          how the stock helmet is placed around the head; reads/writes fit_config.txt
│   ├── helmet_fit.*          the stretch fit, and the "room for lattice" report
│   ├── head_lid.*            which triangles of the head scan are the lid (from the STL header)
│   ├── helmet_cavity.*       "inside the shell": the radius map seen from the head's centre
│   ├── donning.*             the lift-off direction and the swept head (so it can be put on)
│   ├── lattice_generator.*   builds the helmet: all the distance fields, combined and meshed
│   ├── lattice_struts.*      the lattice cells as geometry: re-entrant walls, original strut cell, gyroid
│   ├── mesh_distance.*       fast distances to a triangle mesh (search trees built once)
│   ├── fine_grid.*           block-wise fine sampling grid + marching tetrahedra
│   ├── mesh_simplify.*       simplification with a guaranteed error bound and no self-intersections
│   ├── mesh_components.*     splits the mesh into pieces; removes loose fragments, keeps air pockets
│   ├── print_export.*        the print copy: mm, crown-down, on the bed
│   └── coupon.*              small test prints of the helmet's walls under a roof
├── tools/
│   └── check_output.py       independent checker (Python + Open3D), and the slice pictures
├── data/
│   ├── head.stl              head mesh the program reads (here: GrabCAD's sample head, see below)
│   └── helmet.STL            the stock GrabCAD bike helmet (thin shell with 9 vents)
├── docs/
│   └── DESIGN_NOTES.md       the original design notes (tool choices, first plan)
├── Male_Head_LP.stp          GrabCAD's sample head as CAD (STEP), the source of data/head.stl
├── 4c1d2516-b483-43be-82ae-7a436e8b1888.jpg   photo of the first hand-drawn plan
├── notes.txt                 early project notes (the first plan, in text)
└── build/                    an early Linux build folder, committed by mistake: not source, ignore it
```

---

## Source files

Each `.h` file starts with a comment explaining what it does and why; that is the best place
to read further. Grouped by job:

### Entry point

| File | Purpose |
|---|---|
| `main.cpp` | Parses the command line, loads `data/head.stl` and `data/helmet.STL`, applies the fit (zoom or stretch), runs the generator, writes the outputs. Without `--generate-only` it also opens a 3D viewer (libigl/GLFW) with a small panel to adjust the fit and save it to `fit_config.txt`. `--coupon` makes the test coupons without reading any data. |

### Fitting the stock helmet

| File | Purpose |
|---|---|
| `fit_config.h/.cpp` | The fit as data: **zoom** (Aarav's original: one scale, helmet top a fixed gap above the head) or **stretch** (separate scale per axis, a tilt and an offset). Saves and loads `fit_config.txt` (version 1 = zoom, version 2 = stretch). |
| `helmet_fit.h/.cpp` | Applies the stretch fit to the helmet's vertices (stretch along the helmet's own axes, then tilt, then place). Also measures the room left for lattice around the head, used for the pass/fail gate. |

The *search* for the best stretch fit is a separate Python study (not in this repository, see
below); it writes the chosen numbers into `fit_config.txt`, and this program only applies them.

### Defining the space to fill

| File | Purpose |
|---|---|
| `head_lid.h/.cpp` | The scan of a head is open at the bottom, so it is closed with a "lid". The exporter records the lid's triangles in the STL header (`ME323-HEAD v1 lid-faces <first> <count>`); this file reads and checks that marker. A head without a marker is treated as a complete head. |
| `helmet_cavity.h/.cpp` | "Inside the helmet": for every direction seen from the head's centre (0.5° steps), the distance to the shell's inner face. Vents are filled by an exact harmonic (Laplace) solve; the bottom opening stays open; directions that only see the lid are excluded. Replaces the original inside test, which filled the helmet's whole bounding box. |
| `donning.h/.cpp` | Putting the helmet on: tries straight lift-off directions (tilted up to ±45° front–back, ±10° sideways), picks the one the head blocks least, and gives the distance to the head swept along it, so liner and lattice stay out of the head's path. |

### Building the helmet

| File | Purpose |
|---|---|
| `lattice_generator.h/.cpp` | The core. Builds the distance fields of the shell (the stock wall plus a thin skin, or the original padded band), the liner (3.7 mm against the swept head), the lattice and the cavity, combines them with min/max, and meshes the result: coarse grid (original) or the fine block grid. `GenerationOptions` holds every setting. |
| `lattice_struts.h/.cpp` | The lattice as pure geometry, independent of any grid: the **re-entrant honeycomb** (`ReentrantCell`, `reentrant_wall_distance`: walls extruded along the helmet's up axis, pattern centred on the head's midline), the optional roof chamfer under the liner, Aarav's original strut cell and the gyroid (TPMS) sheet. |
| `mesh_distance.h/.cpp` | Signed/unsigned distance to a triangle mesh at single points, with the search trees built once, so the fine grid can query them millions of times on many threads. |

### From field to triangles

| File | Purpose |
|---|---|
| `fine_grid.h/.cpp` | A sampling grid evaluated and meshed in blocks: only blocks a surface can pass through are sampled finely (0.37 mm), with exact distances near the surface and safe bounds elsewhere. Marching tetrahedra per block, with vertices on shared block faces matched exactly so blocks join without gaps; checks the result is closed. |
| `mesh_simplify.h/.cpp` | Edge-collapse simplification (on libigl's engine, our own rules): every original point stays within the tolerance (0.05 mm) of the result, no collapse may flip a triangle, make a needle or make the surface cross itself, block borders stay fixed, and vertices stay far enough apart to survive saving as 32-bit STL. 118.85 M → 5.65 M triangles. |
| `mesh_components.h/.cpp` | Splits the closed mesh into pieces; removes loose fragments (too small to print), keeps the main body and the sealed air pockets inside it (the honeycomb's closed cells), and reports anything unexpected. |

### Printing

| File | Purpose |
|---|---|
| `print_export.h/.cpp` | The copy for the slicer: converted to mm, turned crown-down by a proper rotation `(x, y, z) → (x, z, −y)` (not a mirror), lowest point on the bed, centred. Crown-down needs the least support inside the part. |
| `coupon.h/.cpp` | Two 40 × 40 × 32.7 mm test prints with the helmet's exact walls between a floor (like the shell) and a roof (like the liner): A with a flat roof bridging the cells, B with 55° slopes under the roof. If A prints cleanly, the helmet can be printed as is; if A's roof sags, use `--roof-chamfer 55`. |

### Checking

| File | Purpose |
|---|---|
| `tools/check_output.py` | Independent checker, deliberately built on Open3D's ray tracer rather than libigl. Sorts every bit of material into shell / liner / lattice / outside and checks: nothing outside the helmet; nothing where the head was not scanned; nothing in the head's way when lifting off; the shell has no holes; the fit's gate (room ≥ 20 mm, evenness, size, coverage). Also reports volume and mass, sealed pockets, overhangs, and draws the side/front slice pictures. |

---

## How a run flows through the code

```
main.cpp
 ├─ read data/head.stl (+ lid marker: head_lid) and data/helmet.STL
 ├─ fit the helmet: zoom, or stretch from fit_config.txt (fit_config, helmet_fit)
 └─ generate_final_helmet()                                  lattice_generator
     ├─ inside the shell: radius map, vents bridged            helmet_cavity
     ├─ unscanned directions removed                           helmet_cavity + head_lid
     ├─ lift-off direction, swept head                         donning
     ├─ fields: shell, liner, lattice walls, cavity            lattice_struts, mesh_distance
     ├─ combine with min/max and sample                        fine_grid (or the coarse grid)
     ├─ marching tetrahedra + simplification per block         fine_grid, mesh_simplify
     ├─ remove loose fragments, re-check closed                mesh_components
     └─ write final_helmet_<case>.stl and *_print.stl          print_export
```

---

## Build

Needs CMake ≥ 3.20 and a C++20 compiler; libigl (with Eigen, GLFW and ImGui) is downloaded by
CMake on the first configure.

```bash
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-local -j 8
```

Use a fresh build folder (the committed `build/` is a stale Linux build). With Ninja installed,
add `-G Ninja`. The checker needs Python 3 with `numpy`, `scipy`, `open3d` and `matplotlib`.

---

## Run

Run from the repository root, because the program reads `data/` and `fit_config.txt` from the
working directory. Sizes on the command line are in the head file's units (metres for our scan,
so `0.0012` = 1.2 mm).

```bash
./build-local/me323 --generate-only --export-fit     # build the helmet, no viewer
./build-local/me323                                  # the same, then open the 3D viewer
./build-local/me323 --coupon                         # just the two test coupons (mm)
./build-local/me323 --classic --generate-only        # Aarav's original behaviour, for comparison
```

**Defaults** (no flags): auxetic re-entrant honeycomb, θ −35°, 10 mm cells, 1.2 mm walls,
0.37 mm surface grid, simplified to 0.05 mm, printable output.

**Outputs** (in the working directory):

| File | What it is |
|---|---|
| `final_helmet_auxetic.stl` | the helmet in the model's units and orientation (used for the checks) |
| `final_helmet_auxetic_print.stl` | **the file to print**: mm, crown-down, on the bed |
| `fitted_head.stl`, `fitted_helmet.stl` | with `--export-fit`: the exact head and helmet the generator used (the checker needs them) |
| `coupon_A_no_chamfer.stl`, `coupon_B_chamfer.stl` | with `--coupon`: the two test prints |

**Useful flags:**

| Flag | Effect |
|---|---|
| `--fit zoom\|stretch` | choose the fit; stretch needs a version-2 `fit_config.txt` |
| `--lattice auxetic\|tpms` | re-entrant honeycomb (default) or a gyroid sheet |
| `--cell original\|reentrant` | Aarav's strut cell or the re-entrant honeycomb |
| `--cell-width`, `--cell-angle`, `--cell-h-ratio`, `--wall-thickness` | the honeycomb's size, angle (negative = re-entrant), h/l (0 = the value giving ν = −1) and wall thickness |
| `--fine-spacing <size>` | surface grid (default 0.0004; 0 = the coarse grid only) |
| `--simplify <tolerance>` | simplification tolerance (default 0.00005 = 0.05 mm; 0 = none) |
| `--roof-chamfer <deg>` | 45–80°: slopes under the liner, so the roof needs no bridging |
| `--shell wall\|band` | shell from the stock wall plus a skin (default) or the original padded band |
| `--classic` | the original behaviour: original cell, coarse grid, no simplification |

---

## Check an output

```bash
python3 tools/check_output.py final_helmet_auxetic.stl fitted_head.stl fitted_helmet.stl \
    --step3 --zone-helmet data/helmet.STL --fine-spacing 0.0004 --overhangs --density 1.24 \
    --png slices.png
```

It prints a PASS/FAIL line per check and exits with status 0 only if all pass. `--density` is
the filament's density in g/cm³ for the mass estimate (1.24 PLA, 1.21 TPU). The slice picture
colours shell, liner and lattice, and marks any failure in red, purple or pink.

For the final helmet: material outside the helmet under 0.005% (the original code: 51%),
nothing in unscanned directions or in the head's way, shell whole, gate passed (room ≥ 20 mm,
p95/p5 ≤ 1.5, size ≤ 245 × 280 mm, coverage ≥ 99%).

---

## What is not in this repository

- **The head scan.** `data/head.stl` here is GrabCAD's sample head, as originally committed.
  Our results use a scan of a real person, which is kept out of the repository on purpose. To use your own scan, replace `data/head.stl` with a closed head mesh whose lid is
  marked in the STL header (see `head_lid.h`); without a marker the head is taken as complete.
- **The fit for our scan.** `fit_config.txt` (version 2, stretch) comes from the fit study
  and is not committed. Without it the program uses the zoom fit.
- **The scan pipeline** (Python: video frames → structure-from-motion with pycolmap → smooth
  head surface → lidded STL), the **stretch-fit search**, the **beam model** that gives the
  Poisson's ratios, the unit tests and the generated STLs (about 280 MB each) are kept in the
  project folder outside this repository; they are available on request.

---

## History

| Commit | Change |
|---|---|
| `e9f2b65`, `ecac326`, `0e578fa` | Aarav: first version — helmet fit (zoom), lattice generation (strut cell, gyroid), viewer |
| `731a8ea` | Step 1: lattice confined to the inside of the helmet (radius map); crash for large gaps fixed |
| `071bda4` | Step 2: bottom opened, lid marker, lift-off direction; the helmet can be put on |
| `ce132eb` | Step 3: stretch fit of the stock helmet; shell from the stock wall plus a thin skin |
| `f08e59d` | Step 4a: fine surface grid, sampled only near the surfaces |
| `3bea3c9` | Step 4b: re-entrant honeycomb walls; simplification with guaranteed error |
| `7af4f73` | Steps 4c–4e: print file (crown-down), test coupons, fragment clean-up, new defaults |

---

## Credits

- **Aarav Jayalwal** — the original generator: fit, signed-distance grid, lattice cells,
  marching tetrahedra, viewer.
- **Vihaan Thaker** — head-scan pipeline and the changes from step 1 on; developed with AI
  assistance (Claude Code).
- **Devansh** — the scan subject.
- Stock helmet and sample head from GrabCAD.
