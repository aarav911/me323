#include "fine_grid.h"

#include "mesh_simplify.h"

#include <Eigen/Geometry>
#include <igl/default_num_threads.h>
#include <igl/parallel_for.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

namespace
{

constexpr std::uint64_t sample_flag = std::uint64_t{1} << 63;
constexpr std::uint64_t not_shared = ~std::uint64_t{0};

// One block's piece of the mesh, before merging.
struct BlockMesh
{
    std::vector<Eigen::RowVector3d> vertices;
    std::vector<std::uint64_t> shared_key;   // not_shared, or the global key
    std::vector<char> at_sample;             // made at a sample exactly at the isovalue
    std::vector<Eigen::Vector3i> faces;      // local vertex indices
    std::size_t samples = 0;
    std::size_t negative = 0;
    SimplifyResult simplify;                 // outcome, when simplification was asked for
};

// Simplifies a block's piece in place (see mesh_blocks()). Locked: every
// vertex shared with another block, and every exact-isovalue sample vertex
// (where two sheets can touch at a point).
void simplify_piece(BlockMesh& piece, double tolerance, const FineGrid& grid, const FineBlock& block)
{
    // The block's own box: the piece must stay inside it (see mesh_simplify.h).
    const Eigen::RowVector3d box_min = grid.point(block.x0, block.y0, block.z0);
    const Eigen::RowVector3d box_max =
        grid.point(block.x0 + block.sx - 1, block.y0 + block.sy - 1, block.z0 + block.sz - 1);
    const Eigen::Index n = static_cast<Eigen::Index>(piece.vertices.size());
    Eigen::MatrixXd V(n, 3);
    std::vector<char> locked(piece.vertices.size(), 0);
    for (Eigen::Index v = 0; v < n; ++v)
    {
        V.row(v) = piece.vertices[static_cast<std::size_t>(v)];
        locked[static_cast<std::size_t>(v)] =
            (piece.shared_key[static_cast<std::size_t>(v)] != not_shared ||
             piece.at_sample[static_cast<std::size_t>(v)]) ? 1 : 0;
    }
    Eigen::MatrixXi F(static_cast<Eigen::Index>(piece.faces.size()), 3);
    for (std::size_t f = 0; f < piece.faces.size(); ++f)
    {
        F.row(static_cast<Eigen::Index>(f)) = piece.faces[f].transpose();
    }
    Eigen::MatrixXd U;
    Eigen::MatrixXi G;
    Eigen::VectorXi I;
    piece.simplify = simplify_mesh(V, F, locked, tolerance, U, G, I, &box_min, &box_max);
    if (piece.simplify.outcome != SimplifyOutcome::Simplified)
    {
        return;
    }
    std::vector<Eigen::RowVector3d> vertices(static_cast<std::size_t>(U.rows()));
    std::vector<std::uint64_t> keys(static_cast<std::size_t>(U.rows()));
    std::vector<char> at_sample(static_cast<std::size_t>(U.rows()));
    for (Eigen::Index u = 0; u < U.rows(); ++u)
    {
        const std::size_t origin = static_cast<std::size_t>(I(u));
        vertices[static_cast<std::size_t>(u)] = U.row(u);
        keys[static_cast<std::size_t>(u)] = piece.shared_key[origin];
        at_sample[static_cast<std::size_t>(u)] = piece.at_sample[origin];
    }
    std::vector<Eigen::Vector3i> faces(static_cast<std::size_t>(G.rows()));
    for (Eigen::Index f = 0; f < G.rows(); ++f)
    {
        faces[static_cast<std::size_t>(f)] = G.row(f).transpose();
    }
    piece.vertices = std::move(vertices);
    piece.shared_key = std::move(keys);
    piece.at_sample = std::move(at_sample);
    piece.faces = std::move(faces);
}

void orient_triangle(
    Eigen::Vector3i& triangle,
    const std::vector<Eigen::RowVector3d>& vertices,
    const Eigen::RowVector3d& inside_center,
    const Eigen::RowVector3d& outside_center)
{
    const Eigen::RowVector3d a = vertices[triangle[0]];
    const Eigen::RowVector3d b = vertices[triangle[1]];
    const Eigen::RowVector3d c = vertices[triangle[2]];
    if ((b - a).cross(c - a).dot(outside_center - inside_center) < 0.0)
    {
        std::swap(triangle[1], triangle[2]);
    }
}

// Marching tetrahedra on one block: the original algorithm, with samples
// addressed by their global index (for vertex keys and positions) and their
// value read from the block's own array.
// end_margin > 0 (printable output): no crossing closer than that fraction
// of an edge to either end of it (see mesh_blocks()).
void march_block(
    const FineGrid& grid,
    const FineBlock& block,
    const std::vector<double>& field,
    double isovalue,
    BlockMesh& out,
    double end_margin)
{
    constexpr int cube_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    constexpr int tetrahedra[6][4] = {
        {0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
        {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}
    };
    const unsigned x1 = block.x0 + block.sx - 1;
    const unsigned y1 = block.y0 + block.sy - 1;
    const unsigned z1 = block.z0 + block.sz - 1;

    struct Corner
    {
        std::size_t global;
        std::size_t local;
        unsigned x, y, z;
    };
    const auto on_boundary = [&](const Corner& c)
    {
        return c.x == block.x0 || c.x == x1 || c.y == block.y0 || c.y == y1 ||
               c.z == block.z0 || c.z == z1;
    };
    // An edge is shared with a neighbouring block only if it lies in one of
    // this block's boundary planes.
    const auto edge_shared = [&](const Corner& a, const Corner& b)
    {
        return (a.x == b.x && (a.x == block.x0 || a.x == x1)) ||
               (a.y == b.y && (a.y == block.y0 || a.y == y1)) ||
               (a.z == b.z && (a.z == block.z0 || a.z == z1));
    };
    const auto position = [&](const Corner& c) { return grid.point(c.x, c.y, c.z); };

    std::unordered_map<std::uint64_t, int> edge_vertices;
    edge_vertices.reserve(block.sample_count() / 8 + 16);

    const auto add_vertex = [&](std::uint64_t key, bool shared,
                                const Eigen::RowVector3d& p) -> int
    {
        const auto [it, inserted] =
            edge_vertices.emplace(key, static_cast<int>(out.vertices.size()));
        if (inserted)
        {
            out.vertices.push_back(p);
            out.shared_key.push_back(shared ? key : not_shared);
            out.at_sample.push_back((key & sample_flag) != 0 ? 1 : 0);
        }
        return it->second;
    };
    const auto vertex_for_edge = [&](const Corner& a, const Corner& b) -> int
    {
        if (field[a.local] == isovalue)
        {
            return add_vertex(sample_flag | a.global, on_boundary(a), position(a));
        }
        if (field[b.local] == isovalue)
        {
            return add_vertex(sample_flag | b.global, on_boundary(b), position(b));
        }
        const std::size_t low = std::min(a.global, b.global);
        const std::size_t high = std::max(a.global, b.global);
        const std::uint64_t key = (static_cast<std::uint64_t>(low) << 32) | high;
        const auto found = edge_vertices.find(key);
        if (found != edge_vertices.end())
        {
            return found->second;
        }
        // The tetrahedra list an edge's ends in either order (inside first in
        // some cases, outside first in others), and the two blocks sharing an
        // edge meet it through different tetrahedra, so a shared edge is
        // always interpolated from its lower-index end: both blocks then make
        // the same vertex to the last bit. Unshared edges keep the original
        // order, so a single block (the coarse path) is unchanged; its shared
        // edges lie on the grid's outer faces, which no surface reaches.
        const bool shared = edge_shared(a, b);
        const Corner& from = (shared && b.global < a.global) ? b : a;
        const Corner& to = (&from == &a) ? b : a;
        double t = (isovalue - field[from.local]) / (field[to.local] - field[from.local]);
        if (end_margin > 0.0)
        {
            // Symmetric, so it gives the same point whichever end it starts from.
            t = std::clamp(t, end_margin, 1.0 - end_margin);
        }
        return add_vertex(key, shared, position(from) + t * (position(to) - position(from)));
    };
    const auto append_oriented = [&](int a, int b, int c,
                                     const Eigen::RowVector3d& inside_center,
                                     const Eigen::RowVector3d& outside_center)
    {
        Eigen::Vector3i triangle(a, b, c);
        if (triangle[0] == triangle[1] || triangle[1] == triangle[2] ||
            triangle[2] == triangle[0])
        {
            return;
        }
        orient_triangle(triangle, out.vertices, inside_center, outside_center);
        out.faces.push_back(triangle);
    };

    out.samples = field.size();
    out.negative = static_cast<std::size_t>(std::count_if(
        field.begin(), field.end(), [&](double v) { return v < isovalue; }));

    for (unsigned k = 0; k + 1 < block.sz; ++k)
    {
        for (unsigned j = 0; j + 1 < block.sy; ++j)
        {
            for (unsigned i = 0; i + 1 < block.sx; ++i)
            {
                std::array<Corner, 8> cube;
                for (int corner = 0; corner < 8; ++corner)
                {
                    const unsigned li = i + cube_offsets[corner][0];
                    const unsigned lj = j + cube_offsets[corner][1];
                    const unsigned lk = k + cube_offsets[corner][2];
                    const unsigned gx = block.x0 + li;
                    const unsigned gy = block.y0 + lj;
                    const unsigned gz = block.z0 + lk;
                    cube[corner] = {grid.index(gx, gy, gz), block.local(li, lj, lk),
                                    gx, gy, gz};
                }

                for (const auto& tetra : tetrahedra)
                {
                    std::array<int, 4> inside;
                    std::array<int, 4> outside;
                    int inside_count = 0;
                    int outside_count = 0;
                    Eigen::RowVector3d inside_center = Eigen::RowVector3d::Zero();
                    Eigen::RowVector3d outside_center = Eigen::RowVector3d::Zero();
                    for (int corner = 0; corner < 4; ++corner)
                    {
                        const int local = tetra[corner];
                        if (field[cube[local].local] < isovalue)
                        {
                            inside[inside_count++] = local;
                            inside_center += position(cube[local]);
                        }
                        else
                        {
                            outside[outside_count++] = local;
                            outside_center += position(cube[local]);
                        }
                    }
                    if (inside_count == 0 || inside_count == 4)
                    {
                        continue;
                    }
                    inside_center /= inside_count;
                    outside_center /= outside_count;

                    if (inside_count == 1)
                    {
                        const int a = vertex_for_edge(cube[inside[0]], cube[outside[0]]);
                        const int b = vertex_for_edge(cube[inside[0]], cube[outside[1]]);
                        const int c = vertex_for_edge(cube[inside[0]], cube[outside[2]]);
                        append_oriented(a, b, c, inside_center, outside_center);
                    }
                    else if (inside_count == 3)
                    {
                        const int a = vertex_for_edge(cube[outside[0]], cube[inside[0]]);
                        const int b = vertex_for_edge(cube[outside[0]], cube[inside[1]]);
                        const int c = vertex_for_edge(cube[outside[0]], cube[inside[2]]);
                        append_oriented(a, b, c, inside_center, outside_center);
                    }
                    else
                    {
                        const int a = vertex_for_edge(cube[inside[0]], cube[outside[0]]);
                        const int b = vertex_for_edge(cube[inside[0]], cube[outside[1]]);
                        const int c = vertex_for_edge(cube[inside[1]], cube[outside[0]]);
                        const int d = vertex_for_edge(cube[inside[1]], cube[outside[1]]);
                        append_oriented(a, b, c, inside_center, outside_center);
                        append_oriented(b, d, c, inside_center, outside_center);
                    }
                }
            }
        }
    }
}

} // namespace

bool make_fine_grid(
    const Eigen::RowVector3d& coarse_minimum, double coarse_spacing,
    unsigned coarse_nx, unsigned coarse_ny, unsigned coarse_nz,
    double requested_spacing, FineGrid& fine, std::string& error)
{
    if (!(coarse_spacing > 0.0) || !(requested_spacing > 0.0) ||
        coarse_nx < 2 || coarse_ny < 2 || coarse_nz < 2)
    {
        error = "Fine grid: invalid coarse grid or spacing.";
        return false;
    }
    const double ratio = coarse_spacing / requested_spacing;
    const double k = std::max(1.0, std::ceil(ratio * (1.0 - 1e-12)));
    if (k > 64.0)
    {
        error = "Fine grid: the requested spacing is over 64 times finer than the "
                "coarse grid.";
        return false;
    }
    fine.k = static_cast<unsigned>(k);
    fine.minimum = coarse_minimum;
    fine.spacing = coarse_spacing / k;
    fine.nx = (coarse_nx - 1) * fine.k + 1;
    fine.ny = (coarse_ny - 1) * fine.k + 1;
    fine.nz = (coarse_nz - 1) * fine.k + 1;
    if (fine.sample_count() >= (std::size_t{1} << 31))
    {
        error = "Fine grid: " + std::to_string(fine.sample_count()) +
            " samples, more than the 2^31 the mesh vertex keys allow; use a "
            "coarser --fine-spacing.";
        return false;
    }
    return true;
}

FineBlock whole_grid_block(const FineGrid& grid)
{
    FineBlock block;
    block.sx = grid.nx;
    block.sy = grid.ny;
    block.sz = grid.nz;
    return block;
}

std::vector<FineBlock> active_blocks(
    const FineGrid& grid, unsigned coarse_nx, unsigned coarse_ny, unsigned coarse_nz,
    const std::vector<char>& coarse_active, unsigned block_cells,
    std::size_t& total_blocks)
{
    const unsigned cx = coarse_nx - 1;
    const unsigned cy = coarse_ny - 1;
    const unsigned cz = coarse_nz - 1;
    const unsigned bx = (cx + block_cells - 1) / block_cells;
    const unsigned by = (cy + block_cells - 1) / block_cells;
    const unsigned bz = (cz + block_cells - 1) / block_cells;
    total_blocks = static_cast<std::size_t>(bx) * by * bz;
    std::vector<FineBlock> blocks;
    for (unsigned kz = 0; kz < bz; ++kz)
    {
        for (unsigned ky = 0; ky < by; ++ky)
        {
            for (unsigned kx = 0; kx < bx; ++kx)
            {
                const unsigned c0[3] = {kx * block_cells, ky * block_cells, kz * block_cells};
                const unsigned c1[3] = {std::min(c0[0] + block_cells, cx),
                                        std::min(c0[1] + block_cells, cy),
                                        std::min(c0[2] + block_cells, cz)};
                bool active = false;
                for (unsigned z = c0[2]; z < c1[2] && !active; ++z)
                {
                    for (unsigned y = c0[1]; y < c1[1] && !active; ++y)
                    {
                        for (unsigned x = c0[0]; x < c1[0] && !active; ++x)
                        {
                            active = coarse_active[x + static_cast<std::size_t>(cx)
                                * (y + static_cast<std::size_t>(cy) * z)] != 0;
                        }
                    }
                }
                if (!active)
                {
                    continue;
                }
                FineBlock block;
                block.x0 = c0[0] * grid.k;
                block.y0 = c0[1] * grid.k;
                block.z0 = c0[2] * grid.k;
                block.sx = (c1[0] - c0[0]) * grid.k + 1;
                block.sy = (c1[1] - c0[1]) * grid.k + 1;
                block.sz = (c1[2] - c0[2]) * grid.k + 1;
                blocks.push_back(block);
            }
        }
    }
    return blocks;
}

bool mesh_blocks(
    const FineGrid& grid,
    const std::vector<FineBlock>& blocks,
    const BlockField& field,
    double isovalue,
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces,
    MeshStatistics& statistics,
    std::string& error,
    double simplify_tolerance)
{
    if (!(simplify_tolerance >= 0.0) || !std::isfinite(simplify_tolerance))
    {
        error = "Internal: invalid simplification tolerance.";
        return false;
    }
    std::vector<BlockMesh> pieces(blocks.size());
    // Blocks differ a lot in work (the lattice ones most), and they lie in
    // runs, so instead of one fixed range per thread (igl::parallel_for's
    // split) each worker takes the next block left until none are. Each
    // block's result depends only on the block, and the merge below goes in
    // block order, so the output is the same whichever thread did what.
    std::atomic<std::size_t> next_block{0};
    const std::size_t workers = std::max<std::size_t>(1, igl::default_num_threads());
    const auto mesh_one = [&](const std::size_t b)
    {
        std::vector<double> values(blocks[b].sample_count());
        field(blocks[b], values);
        if (simplify_tolerance > 0.0)
        {
            // Printable output: no sample within `guard` of the isovalue (no
            // vertex exactly at a sample, where sheets can meet at a point),
            // and below no crossing within 0.1% of an edge of either end. A
            // crossing very near a sample (a value close to the isovalue, or
            // an edge across one of the field's jumps such as the coverage
            // cut) lies nanometres from the crossings on the sample's other
            // edges, and a 32-bit STL then merges those vertices (folded,
            // non-manifold triangles). Both depend only on the samples, so
            // neighbouring blocks still agree.
            const double guard = 1e-3 * grid.spacing;
            for (double& value : values)
            {
                const double offset = value - isovalue;
                if (std::abs(offset) < guard)
                {
                    value = isovalue + (offset < 0.0 ? -guard : guard);
                }
            }
        }
        march_block(grid, blocks[b], values, isovalue, pieces[b],
                    simplify_tolerance > 0.0 ? 1e-3 : 0.0);
        if (simplify_tolerance > 0.0 && !pieces[b].faces.empty())
        {
            simplify_piece(pieces[b], simplify_tolerance, grid, blocks[b]);
        }
    };
    igl::parallel_for(std::min(workers, blocks.size()), [&](const std::size_t)
    {
        for (std::size_t b = next_block++; b < blocks.size(); b = next_block++)
        {
            mesh_one(b);
        }
    }, 1);

    // Merge in block order. A shared vertex made by two blocks must come out
    // identical from both (same samples, same values, same formula).
    statistics = MeshStatistics{};
    statistics.blocks = blocks.size();
    std::size_t vertex_total = 0;
    std::size_t face_total = 0;
    for (const BlockMesh& piece : pieces)
    {
        vertex_total += piece.vertices.size();
        face_total += piece.faces.size();
        statistics.samples += piece.samples;
        statistics.negative_samples += piece.negative;
        if (simplify_tolerance > 0.0 && !piece.faces.empty())
        {
            const SimplifyOutcome outcome = piece.simplify.outcome;
            statistics.simplified_pieces += outcome == SimplifyOutcome::Simplified;
            statistics.small_pieces += outcome == SimplifyOutcome::TooSmall;
            statistics.refused_pieces += outcome == SimplifyOutcome::Refused;
            statistics.faces_before += piece.simplify.faces_before;
            statistics.faces_after += piece.faces.size();
        }
    }
    std::vector<Eigen::RowVector3d> merged;
    merged.reserve(vertex_total);
    std::vector<Eigen::Vector3i> merged_faces;
    merged_faces.reserve(face_total);
    std::unordered_map<std::uint64_t, int> shared;
    std::size_t mismatches = 0;
    for (BlockMesh& piece : pieces)
    {
        std::vector<int> map(piece.vertices.size());
        for (std::size_t v = 0; v < piece.vertices.size(); ++v)
        {
            if (piece.shared_key[v] == not_shared)
            {
                map[v] = static_cast<int>(merged.size());
                merged.push_back(piece.vertices[v]);
                continue;
            }
            const auto [it, inserted] =
                shared.emplace(piece.shared_key[v], static_cast<int>(merged.size()));
            if (inserted)
            {
                merged.push_back(piece.vertices[v]);
            }
            else if (merged[static_cast<std::size_t>(it->second)] != piece.vertices[v])
            {
                ++mismatches;
            }
            map[v] = it->second;
        }
        for (const Eigen::Vector3i& f : piece.faces)
        {
            merged_faces.emplace_back(map[static_cast<std::size_t>(f[0])],
                                      map[static_cast<std::size_t>(f[1])],
                                      map[static_cast<std::size_t>(f[2])]);
        }
        piece = BlockMesh{};    // merged: free it now, not at the end
    }
    if (mismatches > 0)
    {
        error = "Internal: " + std::to_string(mismatches) + " vertices shared between "
                "blocks came out different in each block.";
        return false;
    }
    if (merged.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        error = "The mesh has more vertices than an int index allows.";
        return false;
    }
    vertices.resize(static_cast<Eigen::Index>(merged.size()), 3);
    for (std::size_t v = 0; v < merged.size(); ++v)
    {
        vertices.row(static_cast<Eigen::Index>(v)) = merged[v];
    }
    faces.resize(static_cast<Eigen::Index>(merged_faces.size()), 3);
    for (std::size_t f = 0; f < merged_faces.size(); ++f)
    {
        faces.row(static_cast<Eigen::Index>(f)) = merged_faces[f];
    }
    return true;
}

bool closed_two_manifold(
    const Eigen::MatrixXi& faces,
    const Eigen::MatrixXd& vertices,
    const Eigen::RowVector3d& grid_min,
    const Eigen::RowVector3d& grid_max,
    double spacing,
    std::size_t& boundary_edges,
    std::size_t& nonmanifold_edges,
    std::size_t& boundary_grid_edges)
{
    std::vector<std::uint64_t> keys;
    keys.reserve(static_cast<std::size_t>(faces.rows()) * 3);
    for (Eigen::Index face = 0; face < faces.rows(); ++face)
    {
        for (int edge = 0; edge < 3; ++edge)
        {
            const std::uint32_t a = static_cast<std::uint32_t>(faces(face, edge));
            const std::uint32_t b = static_cast<std::uint32_t>(faces(face, (edge + 1) % 3));
            keys.push_back((static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b));
        }
    }
    std::sort(keys.begin(), keys.end());
    boundary_edges = 0;
    nonmanifold_edges = 0;
    boundary_grid_edges = 0;
    for (std::size_t i = 0; i < keys.size();)
    {
        std::size_t j = i + 1;
        while (j < keys.size() && keys[j] == keys[i])
        {
            ++j;
        }
        const std::size_t count = j - i;
        if (count == 1)
        {
            ++boundary_edges;
            const auto a = static_cast<Eigen::Index>(keys[i] >> 32);
            const auto b = static_cast<Eigen::Index>(keys[i] & 0xffffffffu);
            for (const Eigen::Index vertex : {a, b})
            {
                const Eigen::RowVector3d p = vertices.row(vertex);
                if ((p - grid_min).minCoeff() < spacing * 1e-5 ||
                    (grid_max - p).minCoeff() < spacing * 1e-5)
                {
                    ++boundary_grid_edges;
                    break;
                }
            }
        }
        else if (count != 2)
        {
            ++nonmanifold_edges;
        }
        i = j;
    }
    return boundary_edges == 0 && nonmanifold_edges == 0;
}
