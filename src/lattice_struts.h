#pragma once

#include <Eigen/Core>

#include <cstddef>
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
