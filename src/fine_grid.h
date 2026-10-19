#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// A sampling grid evaluated and meshed in blocks, so only the blocks a surface
// can pass through are ever evaluated (the narrow band around the helmet
// layer), with memory bounded by a block rather than the whole grid. The
// coarse path uses it too, as a single block covering the whole grid.
//
// Sample (x, y, z) sits at minimum + spacing * (x, y, z); global sample index
// x + nx * (y + ny * z). Indices must stay below 2^31: mesh vertex keys pack
// two of them into 64 bits and use the top bit as a flag.
struct FineGrid
{
    Eigen::RowVector3d minimum = Eigen::RowVector3d::Zero();
    double spacing = 0.0;
    unsigned nx = 0;
    unsigned ny = 0;
    unsigned nz = 0;
    unsigned k = 1;                 // samples per coarse cell (fine spacing = coarse / k)

    std::size_t sample_count() const
    {
        return static_cast<std::size_t>(nx) * ny * nz;
    }
    std::size_t index(unsigned x, unsigned y, unsigned z) const
    {
        return static_cast<std::size_t>(x) + static_cast<std::size_t>(nx)
            * (static_cast<std::size_t>(y) + static_cast<std::size_t>(ny) * z);
    }
    Eigen::RowVector3d point(unsigned x, unsigned y, unsigned z) const
    {
        return minimum + spacing * Eigen::RowVector3d(x, y, z);
    }
};

// Fine grid over a coarse grid (same minimum, coarse spacing / k, k the
// smallest integer with spacing <= requested). Fails if it would exceed the
// index limit.
bool make_fine_grid(
    const Eigen::RowVector3d& coarse_minimum, double coarse_spacing,
    unsigned coarse_nx, unsigned coarse_ny, unsigned coarse_nz,
    double requested_spacing, FineGrid& fine, std::string& error);

// Samples x0 .. x0 + sx - 1 (etc.) of a grid: its cubes are the sx-1 by sy-1
// by sz-1 cubes between them. Neighbouring blocks share their boundary samples.
struct FineBlock
{
    unsigned x0 = 0, y0 = 0, z0 = 0;
    unsigned sx = 0, sy = 0, sz = 0;

    std::size_t sample_count() const
    {
        return static_cast<std::size_t>(sx) * sy * sz;
    }
    std::size_t local(unsigned i, unsigned j, unsigned k) const
    {
        return static_cast<std::size_t>(i) + static_cast<std::size_t>(sx)
            * (static_cast<std::size_t>(j) + static_cast<std::size_t>(sy) * k);
    }
};

// The whole grid as one block.
FineBlock whole_grid_block(const FineGrid& grid);

// Blocks of `block_cells` coarse cells per side whose coarse cells are marked
// active (coarse_active: one flag per coarse cell, (cnx-1)*(cny-1)*(cnz-1),
// x fastest). Also returns how many blocks there are in all.
std::vector<FineBlock> active_blocks(
    const FineGrid& grid, unsigned coarse_nx, unsigned coarse_ny, unsigned coarse_nz,
    const std::vector<char>& coarse_active, unsigned block_cells,
    std::size_t& total_blocks);

// Fills `values` (block.sample_count(), block-local order) with the field at
// the block's samples. Called from several threads at once.
using BlockField = std::function<void(const FineBlock&, std::vector<double>&)>;

struct MeshStatistics
{
    std::size_t blocks = 0;
    std::size_t samples = 0;
    std::size_t negative_samples = 0;   // field < isovalue
};

// Marching tetrahedra over the blocks (in parallel), merged in block order so
// the result is the same on every run. Each cube is split into the same six
// tetrahedra everywhere, so neighbouring blocks agree on their shared faces;
// vertices on edges lying in a block's boundary planes are keyed by global
// sample indices and shared between the blocks, and interpolated from their
// lower-index end so both blocks make them identically. One block over the
// whole grid gives the same mesh, triangle for triangle and bit for bit, as
// the original single-grid marching_tetrahedra().
bool mesh_blocks(
    const FineGrid& grid,
    const std::vector<FineBlock>& blocks,
    const BlockField& field,
    double isovalue,
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces,
    MeshStatistics& statistics,
    std::string& error);

// Every edge used by exactly two triangles? Sorts the edge keys instead of
// hashing them (same counts, a fraction of the memory for large meshes).
// boundary_grid_edges counts open edges touching the grid's bounds.
bool closed_two_manifold(
    const Eigen::MatrixXi& faces,
    const Eigen::MatrixXd& vertices,
    const Eigen::RowVector3d& grid_min,
    const Eigen::RowVector3d& grid_max,
    double spacing,
    std::size_t& boundary_edges,
    std::size_t& nonmanifold_edges,
    std::size_t& boundary_grid_edges);
