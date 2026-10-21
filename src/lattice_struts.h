#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

// The lattice as geometry, independent of any sampling grid, so the coarse
// grid and the fine narrow band (fine_grid.h) draw exactly the same struts.

// One strut: a capsule of `radius` around the segment a-b.
struct Strut
{
    Eigen::Vector3d a;
    Eigen::Vector3d b;
};

// Aarav's re-entrant cell, repeated on a square grid anchored at `anchor`,
// covering [grid_min, grid_max] with a cell of slack on every side: the same
// cells, in the same order, as the original build_auxetic_field().
std::vector<Strut> auxetic_struts(
    const Eigen::RowVector3d& grid_min,
    const Eigen::RowVector3d& grid_max,
    double cell_size,
    const Eigen::RowVector3d& anchor);

double auxetic_strut_radius(double cell_size);

// Distance from p to the segment start-end.
double point_segment_distance(
    const Eigen::Vector3d& point,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end);

// Signed distance to the gyroid sheet (approximate, as originally defined).
double gyroid_distance(
    const Eigen::RowVector3d& point,
    double cell_size,
    const Eigen::RowVector3d& anchor);

// A box of samples of a regular grid: sample (i, j, k) of the box sits at
// origin + spacing * (first + (i, j, k)), stored i fastest.
struct SampleBox
{
    Eigen::RowVector3d origin = Eigen::RowVector3d::Zero();   // the grid's minimum
    double spacing = 0.0;
    long first[3] = {0, 0, 0};    // grid index of the box's first sample
    long count[3] = {0, 0, 0};    // samples along each axis

    Eigen::Vector3d point(long i, long j, long k) const
    {
        return (origin + spacing * Eigen::RowVector3d(
            static_cast<double>(first[0] + i), static_cast<double>(first[1] + j),
            static_cast<double>(first[2] + k))).transpose();
    }
};

// Lowers values[...] to the distance from each sample to the strut's surface,
// for the samples within the strut's bounding box widened by `pad` (>= radius).
// With pad = radius this is exactly the original rasterize_strut(). Points
// beyond the pad are left as they are, so pad must be at least radius plus a
// sample diagonal wherever a surface is extracted from the values.
template <typename Value>
void rasterize_strut(
    const SampleBox& box,
    const Strut& strut,
    double radius,
    double pad,
    std::vector<Value>& values);

extern template void rasterize_strut<float>(
    const SampleBox&, const Strut&, double, double, std::vector<float>&);
extern template void rasterize_strut<double>(
    const SampleBox&, const Strut&, double, double, std::vector<double>&);

// The corrected auxetic lattice (step 4b): the textbook re-entrant honeycomb
// (Gibson & Ashby; every other row shifted by half a repeat) drawn in the
// horizontal XZ plane and extruded straight up (Y) as continuous walls, which
// an FDM printer builds layer on layer with no overhangs. In XZ, relative to
// the anchor, a node sits at column c and row r wherever c and r have the same
// parity: the bottom node (c dx, r dz). From it a wall of length h goes up +Z
// to the top node (c dx, r dz + h), and from there two walls of length l slant
// at `angle` (negative = re-entrant) down to the next row's bottom nodes at
// columns c - 1 and c + 1. dx = l cos(angle), dz = h + l sin(angle); the repeat
// is 2 dx (= width) across X and 2 dz along Z. Same layout as textbook() in
// analysis_tools/poisson_frame.py, whose beam model gives this cell's nu.
struct ReentrantCell
{
    double width = 0.0;            // repeat across X
    double angle_degrees = 0.0;    // theta, negative
    double h_ratio = 0.0;          // h / l
    double wall_thickness = 0.0;
    double l = 0.0;
    double h = 0.0;
    double dx = 0.0;               // l cos(theta): half the repeat across X
    double dz = 0.0;               // h + l sin(theta): row spacing along Z
    double slant_dz = 0.0;         // l sin(theta) (< 0)
    double centreline_gap = 0.0;   // closest approach of walls that share no node
    double clear_gap = 0.0;        // centreline_gap - wall_thickness
    // Radius of the largest empty circle between the walls (from their
    // surfaces): how far a roof chamfer must reach to close a cell's roof.
    // An upper bound, within a fiftieth of a row's spacing.
    double largest_opening = 0.0;
};

// h / l giving Gibson-Ashby nu = -1 both ways: cos^2(t) / (-sin t) - sin t.
double reentrant_ideal_h_ratio(double angle_degrees);

// Derives the cell and checks it: finite values, -60 <= angle <= -10, width
// and wall > 0, rows that advance (dz > 0), and walls that do not meet at
// least a wall thickness apart (brute force over a patch). h_ratio 0 means the
// ideal value.
bool make_reentrant_cell(
    double width, double angle_degrees, double h_ratio, double wall_thickness,
    ReentrantCell& cell, std::string& error);

// Gibson-Ashby (thin-wall) Poisson's ratios: squeezed along X, and along Z.
void reentrant_ideal_poisson(const ReentrantCell& cell, double& nu_x, double& nu_z);

// Wall centrelines (XZ, relative to the anchor) of the nodes in rows r0..r1,
// columns c0..c1 (same parity only): each node's upright wall and its two
// slanted walls. Each wall belongs to exactly one node, so none repeats.
std::vector<std::array<Eigen::Vector2d, 2>> reentrant_segments(
    const ReentrantCell& cell, int r0, int r1, int c0, int c1);

// Distance from p to the nearest wall surface (negative inside a wall): the XZ
// distance to the wall centrelines minus half the wall thickness, the same at
// every height. Exact wherever the centreline distance is below `reach`;
// elsewhere at least reach - wall / 2. A pure function of p.
// A chamfer under a roof (step 4d), for printing a roof over open cells
// without bridging: a sloped face rising at `angle_degrees` from horizontal,
// from the walls up to the roof, meeting the roof `reach` away from the
// walls (and the walls reach * tan(angle) below the roof). `wall` and
// `roof` are the distances to the wall and roof surfaces (positive in the
// open cell); the result is negative in the chamfer. Take the minimum with
// the walls and the roof for the solid. Steeper than 45 degrees, the slope
// needs no support.
double roof_chamfer(double wall, double roof, double angle_degrees, double reach);

double reentrant_wall_distance(
    const Eigen::RowVector3d& p,
    const ReentrantCell& cell,
    const Eigen::RowVector3d& anchor,
    double reach);
