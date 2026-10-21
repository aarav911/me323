#include "lattice_generator.h"

#include "donning.h"
#include "fine_grid.h"
#include "helmet_cavity.h"
#include "helmet_fit.h"
#include "lattice_struts.h"
#include "mesh_distance.h"

#include <igl/parallel_for.h>

#include <igl/signed_distance.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace
{

constexpr std::size_t max_grid_samples = 4'000'000;

// Formats without touching std::cout's own flags, which later output relies on.
std::string fixed1(double value)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f", value);
    return text;
}

std::string describe_lift_off(const LiftOff& lift)
{
    if (lift.tilt_degrees == 0.0 && lift.side_degrees == 0.0)
    {
        return "straight up";
    }
    std::string text = "up, tilted";
    if (lift.tilt_degrees != 0.0)
    {
        text += " " + fixed1(std::abs(lift.tilt_degrees)) + " deg toward "
            + (lift.tilt_degrees < 0.0 ? "-Z (the back of the head)" : "+Z (the face)");
    }
    if (lift.side_degrees != 0.0)
    {
        text += std::string(lift.tilt_degrees != 0.0 ? " and" : "") + " "
            + fixed1(std::abs(lift.side_degrees)) + " deg toward "
            + (lift.side_degrees < 0.0 ? "-X" : "+X");
    }
    return text;
}

struct Grid
{
    Eigen::RowVector3d minimum;
    Eigen::RowVector3d step;
    unsigned nx = 0;
    unsigned ny = 0;
    unsigned nz = 0;

    std::size_t index(unsigned x, unsigned y, unsigned z) const
    {
        return static_cast<std::size_t>(x)
            + static_cast<std::size_t>(nx) * y
            + static_cast<std::size_t>(nx) * ny * z;
    }

    Eigen::RowVector3d point(unsigned x, unsigned y, unsigned z) const
    {
        return minimum + step.cwiseProduct(
            Eigen::RowVector3d(x, y, z));
    }

    std::size_t sample_count() const
    {
        return static_cast<std::size_t>(nx) * ny * nz;
    }
};

Eigen::RowVector3d grid_point(const Grid& grid, std::size_t index)
{
    const unsigned x = static_cast<unsigned>(index % grid.nx);
    const unsigned y = static_cast<unsigned>((index / grid.nx) % grid.ny);
    const unsigned z = static_cast<unsigned>(
        index / (static_cast<std::size_t>(grid.nx) * grid.ny));
    return grid.point(x, y, z);
}

bool compute_signed_distance(
    const Eigen::MatrixXd& points,
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    Eigen::VectorXd& distances,
    std::string& error,
    const char* label)
{
    Eigen::VectorXi closest_faces;
    Eigen::MatrixXd closest_points;
    Eigen::MatrixXd closest_normals;
    igl::signed_distance(
        points, vertices, faces, igl::SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        distances, closest_faces, closest_points, closest_normals);
    if (distances.size() != points.rows() || !distances.allFinite())
    {
        error = std::string("Could not compute a finite signed-distance field for the ")
            + label + " mesh.";
        return false;
    }
    return true;
}

// Unsigned on purpose. The helmet is an open shell (rim, vents), so the side of
// it a point lies on is not a meaningful question; the old pseudonormal sign
// was also unreliable on the unwelded STL triangle soup. The shell skin only
// needs the distance. Inside/outside comes from HelmetCavity instead.
bool compute_unsigned_distance(
    const Eigen::MatrixXd& points,
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    Eigen::VectorXd& distances,
    std::string& error,
    const char* label)
{
    Eigen::VectorXi closest_faces;
    Eigen::MatrixXd closest_points;
    Eigen::MatrixXd unused_normals;   // not filled in UNSIGNED mode
    igl::signed_distance(
        points, vertices, faces, igl::SIGNED_DISTANCE_TYPE_UNSIGNED,
        distances, closest_faces, closest_points, unused_normals);
    if (distances.size() != points.rows() || !distances.allFinite())
    {
        error = std::string("Could not compute a finite distance field for the ")
            + label + " mesh.";
        return false;
    }
    return true;
}

void log_simplification(const MeshStatistics& statistics, double tolerance)
{
    if (!(tolerance > 0.0))
    {
        return;
    }
    const double kept = statistics.faces_before > 0
        ? 100.0 * static_cast<double>(statistics.faces_after) / statistics.faces_before : 100.0;
    std::cout << "  simplified to within " << tolerance << ": " << statistics.faces_before
              << " -> " << statistics.faces_after << " triangles (" << fixed1(kept)
              << "% kept); pieces simplified " << statistics.simplified_pieces
              << ", too small " << statistics.small_pieces << ", refused (left as extracted) "
              << statistics.refused_pieces << ".\n";
}

// What the fine surface needs from the coarse pass.
struct FineContext
{
    const Grid& grid;
    const FineGrid& fine;
    const GenerationOptions& options;
    const Eigen::MatrixXd& head_vertices;
    const Eigen::MatrixXi& head_faces;
    const Eigen::MatrixXd& helmet_vertices;
    const Eigen::MatrixXi& helmet_faces;
    const HelmetCavity& cavity;
    const LiftOff& lift;
    const Eigen::VectorXd& head_distance;
    const std::vector<double>& shell_field;      // coarse samples, with the fine skin
    const std::vector<double>& swept;
    const std::vector<char>& swept_computed;
    const std::vector<double>& inside_helmet;
    const std::vector<char>& covered_near;
    Eigen::RowVector3d head_min;
    Eigen::RowVector3d head_max;
    Eigen::RowVector3d grid_last;
    LatticeType lattice_type;
    const ReentrantCell* reentrant;              // null: the lattice is the original auxetic or the gyroid
    double cell_size;
    Eigen::RowVector3d lattice_anchor;
    double shell_thickness;
    double wall_skin;
    double liner_center_distance;
    double liner_thickness;
    double head_clearance;
    double cavity_margin;
};

// The final solid, sampled on the fine grid in the blocks a surface can pass
// through, combined exactly as the coarse path does (shell, liner, clipped
// lattice). Distances are taken from the meshes, exactly, wherever a value can
// be near zero; elsewhere a bound with the right sign stands in. All three
// distance fields change by at most the distance moved (the swept head's is
// the distance to the swept volume), so if every corner of the coarse cell
// around a sample is further than a cell diagonal plus a margin from zero, the
// value at the sample is too, on the same side.
bool generate_fine_surface(const FineContext& c, GeneratedMesh& output, std::string& error)
{
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    const Grid& grid = c.grid;
    const FineGrid& fine = c.fine;
    const double v = grid.step.x();
    const double diagonal = std::sqrt(3.0) * v;
    // A bound stands in for a distance only where it is at least v from zero
    // (well clear of the isovalue): the value at a sample is within `reach`
    // of every corner of its coarse cell, so a bound is taken only where the
    // corners are `reach + v` clear, and the bound itself is corner - reach.
    const double near = diagonal + v;                    // shell: reach = diagonal
    const double swept_reach = diagonal + 0.25 * v;      // + the coarse sweep's own error
    const double swept_near = swept_reach + v;
    const double liner_outer = c.liner_center_distance + c.liner_thickness * 0.5;
    const double sweep_margin = 7.0 * v;                 // as for the coarse sweep
    const Eigen::RowVector3d sweep_box_min = c.head_min.array() - sweep_margin;
    const Eigen::RowVector3d sweep_box_max = c.head_max.array() + sweep_margin;
    const bool band = c.options.shell_mode == ShellMode::Band;

    // Coarse cells that can hold a surface of the solid, grown by one cell.
    const unsigned cx = grid.nx - 1;
    const unsigned cy = grid.ny - 1;
    const unsigned cz = grid.nz - 1;
    const auto cell_index = [&](unsigned x, unsigned y, unsigned z)
    {
        return static_cast<std::size_t>(x) + static_cast<std::size_t>(cx)
            * (static_cast<std::size_t>(y) + static_cast<std::size_t>(cy) * z);
    };
    std::vector<char> cell_active(static_cast<std::size_t>(cx) * cy * cz, 0);
    for (unsigned z = 0; z < cz; ++z)
    {
        for (unsigned y = 0; y < cy; ++y)
        {
            for (unsigned x = 0; x < cx; ++x)
            {
                bool any = c.options.fine_all_blocks;
                for (int corner = 0; corner < 8 && !any; ++corner)
                {
                    const std::size_t i = grid.index(
                        x + (corner & 1), y + ((corner >> 1) & 1), z + ((corner >> 2) & 1));
                    if (std::abs(c.shell_field[i]) <= near)
                    {
                        any = true;                                  // shell
                    }
                    else if (c.covered_near[i] &&
                             (c.swept[i] <= liner_outer + swept_near ||   // liner
                              (c.inside_helmet[i] < 3.0 * v &&            // lattice
                               c.swept[i] >= c.head_clearance - swept_near)))
                    {
                        any = true;
                    }
                }
                cell_active[cell_index(x, y, z)] = any ? 1 : 0;
            }
        }
    }
    std::vector<char> cell_grown(cell_active.size(), 0);
    for (unsigned z = 0; z < cz; ++z)
    {
        for (unsigned y = 0; y < cy; ++y)
        {
            for (unsigned x = 0; x < cx; ++x)
            {
                if (!cell_active[cell_index(x, y, z)])
                {
                    continue;
                }
                for (unsigned zz = (z > 0 ? z - 1 : 0); zz <= std::min(z + 1, cz - 1); ++zz)
                {
                    for (unsigned yy = (y > 0 ? y - 1 : 0); yy <= std::min(y + 1, cy - 1); ++yy)
                    {
                        for (unsigned xx = (x > 0 ? x - 1 : 0); xx <= std::min(x + 1, cx - 1); ++xx)
                        {
                            cell_grown[cell_index(xx, yy, zz)] = 1;
                        }
                    }
                }
            }
        }
    }
    std::size_t total_blocks = 0;
    const std::vector<FineBlock> blocks = active_blocks(
        fine, grid.nx, grid.ny, grid.nz, cell_grown, c.options.fine_block_cells, total_blocks);
    std::size_t evaluated = 0;
    for (const FineBlock& block : blocks)
    {
        evaluated += block.sample_count();
    }
    constexpr std::size_t max_evaluated = 400'000'000;
    std::cout << "Fine surface grid: spacing " << fine.spacing << " (coarse / " << fine.k
              << "), " << fine.nx << " x " << fine.ny << " x " << fine.nz << "; "
              << blocks.size() << " of " << total_blocks << " blocks hold the helmet layer ("
              << evaluated << " samples to evaluate).\n";
    if (evaluated > max_evaluated)
    {
        error = "The fine grid would evaluate " + std::to_string(evaluated) +
            " samples (limit " + std::to_string(max_evaluated) + "); use a coarser "
            "--fine-spacing.";
        return false;
    }

    MeshDistance head_mesh;
    MeshDistance helmet_mesh;
    if (!head_mesh.init(c.head_vertices, c.head_faces, true, error) ||
        !helmet_mesh.init(c.helmet_vertices, c.helmet_faces, !band, error))
    {
        return false;
    }
    GridField coarse_head;
    coarse_head.minimum = grid.minimum;
    coarse_head.spacing = v;
    coarse_head.nx = static_cast<int>(grid.nx);
    coarse_head.ny = static_cast<int>(grid.ny);
    coarse_head.nz = static_cast<int>(grid.nz);
    coarse_head.values = c.head_distance.data();

    // Struts, each with its box widened by the pad: past the radius by more
    // than a fine sample diagonal, so every sample next to a strut's surface
    // gets its true distance (not "far"), which marching needs on both sides.
    // A block takes a strut when the strut's sample range (computed exactly as
    // rasterize_strut() computes it, floor/ceil included) meets the block's:
    // then a sample shared by two blocks gets the same struts in both, and the
    // two blocks agree on its value (and so on the shared mesh vertices).
    std::vector<Strut> struts;
    std::vector<std::array<long, 6>> strut_range;   // low x, y, z, high x, y, z
    const double radius = auxetic_strut_radius(c.cell_size);
    const double pad = radius + 2.0 * fine.spacing;
    const bool original_auxetic = c.lattice_type == LatticeType::Auxetic && c.reentrant == nullptr;
    if (original_auxetic)
    {
        struts = auxetic_struts(grid.minimum, c.grid_last, c.cell_size, c.lattice_anchor);
        strut_range.reserve(struts.size());
        for (const Strut& strut : struts)
        {
            const Eigen::Vector3d low = strut.a.cwiseMin(strut.b).array() - pad;
            const Eigen::Vector3d high = strut.a.cwiseMax(strut.b).array() + pad;
            std::array<long, 6> range;
            for (int axis = 0; axis < 3; ++axis)
            {
                range[axis] = static_cast<long>(std::floor(
                    (low[axis] - fine.minimum[axis]) / fine.spacing));
                range[3 + axis] = static_cast<long>(std::ceil(
                    (high[axis] - fine.minimum[axis]) / fine.spacing));
            }
            strut_range.push_back(range);
        }
    }

    // Walls: exact within this centreline distance, at least reach - wall/2
    // beyond. That is more than a tetrahedron edge (sqrt 3 spacing) outside any
    // wall, so an edge never joins an inside sample to a capped one.
    const double wall_reach = c.reentrant != nullptr
        ? 0.5 * c.reentrant->wall_thickness + 3.0 * fine.spacing : 0.0;

    std::atomic<std::size_t> exact_shell{0};
    std::atomic<std::size_t> exact_swept{0};
    std::atomic<std::size_t> lattice_points{0};
    const BlockField field = [&](const FineBlock& block, std::vector<double>& values)
    {
        std::vector<double> lattice(block.sample_count(), std::numeric_limits<double>::infinity());
        if (original_auxetic)
        {
            SampleBox box;
            box.origin = fine.minimum;
            box.spacing = fine.spacing;
            box.first[0] = block.x0;
            box.first[1] = block.y0;
            box.first[2] = block.z0;
            box.count[0] = block.sx;
            box.count[1] = block.sy;
            box.count[2] = block.sz;
            for (std::size_t s = 0; s < struts.size(); ++s)
            {
                const std::array<long, 6>& range = strut_range[s];
                bool overlaps = true;
                for (int axis = 0; axis < 3; ++axis)
                {
                    overlaps = overlaps && range[axis] <= box.first[axis] + box.count[axis] - 1 &&
                               range[3 + axis] >= box.first[axis];
                }
                if (overlaps)
                {
                    rasterize_strut<double>(box, struts[s], radius, pad, lattice);
                }
            }
        }
        std::size_t local_shell = 0;
        std::size_t local_swept = 0;
        std::size_t local_lattice = 0;
        for (unsigned k = 0; k < block.sz; ++k)
        {
            for (unsigned j = 0; j < block.sy; ++j)
            {
                for (unsigned i = 0; i < block.sx; ++i)
                {
                    const unsigned gx = block.x0 + i;
                    const unsigned gy = block.y0 + j;
                    const unsigned gz = block.z0 + k;
                    const Eigen::RowVector3d p = fine.point(gx, gy, gz);
                    const std::size_t local = block.local(i, j, k);
                    if (c.reentrant != nullptr)
                    {
                        lattice[local] = reentrant_wall_distance(
                            p, *c.reentrant, c.lattice_anchor, wall_reach);
                    }
                    else if (c.lattice_type == LatticeType::Gyroid)
                    {
                        lattice[local] = gyroid_distance(p, c.cell_size, c.lattice_anchor);
                    }
                    // The coarse cell around the sample (integer arithmetic:
                    // the fine grid is the coarse one subdivided k times).
                    const unsigned qx = std::min(gx / fine.k, grid.nx - 2);
                    const unsigned qy = std::min(gy / fine.k, grid.ny - 2);
                    const unsigned qz = std::min(gz / fine.k, grid.nz - 2);
                    std::size_t corner[8];
                    for (int n = 0; n < 8; ++n)
                    {
                        corner[n] = grid.index(qx + (n & 1), qy + ((n >> 1) & 1),
                                               qz + ((n >> 2) & 1));
                    }

                    // Shell.
                    double low = std::numeric_limits<double>::infinity();
                    double high = -std::numeric_limits<double>::infinity();
                    double closest = std::numeric_limits<double>::infinity();
                    for (const std::size_t n : corner)
                    {
                        low = std::min(low, c.shell_field[n]);
                        high = std::max(high, c.shell_field[n]);
                        closest = std::min(closest, std::abs(c.shell_field[n]));
                    }
                    double shell;
                    if (c.options.fine_exact || closest <= near || (low < 0.0 && high > 0.0))
                    {
                        const double d = helmet_mesh(p);
                        shell = band ? std::abs(d) - c.shell_thickness * 0.5 : d - c.wall_skin;
                        ++local_shell;
                    }
                    else
                    {
                        shell = low > 0.0 ? low - diagonal : high + diagonal;
                    }

                    // Liner and lattice exist only in covered directions.
                    double value;
                    if (!c.cavity.covers(p))
                    {
                        value = std::min(shell, c.cavity.open_distance);
                    }
                    else
                    {
                        const double inside = c.cavity.radial_distance(p, c.cavity_margin);
                        double smallest = std::numeric_limits<double>::infinity();
                        bool all_computed = true;
                        for (const std::size_t n : corner)
                        {
                            smallest = std::min(smallest, c.swept[n]);
                            all_computed = all_computed && c.swept_computed[n];
                        }
                        double swept;
                        if (c.options.fine_exact || !all_computed ||
                            smallest <= liner_outer + swept_near)
                        {
                            swept = swept_head_distance_at(
                                coarse_head, c.lift.direction, sweep_box_min, sweep_box_max,
                                sweep_margin, fine.spacing, p, head_mesh(p));
                            ++local_swept;
                        }
                        else
                        {
                            // A lower bound (the coarse sweep overestimates by at
                            // most a quarter cell), at least v beyond the liner.
                            swept = smallest - swept_reach;
                        }
                        const double inner_liner =
                            std::abs(swept - c.liner_center_distance) - c.liner_thickness * 0.5;
                        const double clipped_lattice =
                            std::max({lattice[local], inside, -swept + c.head_clearance});
                        if (clipped_lattice < 0.0)
                        {
                            ++local_lattice;
                        }
                        value = std::min({shell, inner_liner, clipped_lattice});
                    }
                    values[local] = value;
                }
            }
        }
        exact_shell += local_shell;
        exact_swept += local_swept;
        lattice_points += local_lattice;
    };

    const double isovalue = fine.spacing * 0.05;
    MeshStatistics statistics;
    if (!mesh_blocks(fine, blocks, field, isovalue, output.vertices, output.faces,
                     statistics, error, c.options.simplify_tolerance))
    {
        return false;
    }
    log_simplification(statistics, c.options.simplify_tolerance);
    const double seconds =
        std::chrono::duration<double>(clock::now() - start).count();
    // Counted per block: a sample on a face between blocks counts in each.
    std::cout << "  block samples (shared faces counted per block): " << statistics.samples
              << "; exact distances: shell at " << exact_shell.load() << ", swept head at "
              << exact_swept.load() << "; lattice material at " << lattice_points.load()
              << "; " << output.faces.rows() << " triangles; "
              << fixed1(seconds) << " s.\n";
    if (lattice_points.load() == 0)
    {
        error = "No lattice fits between the fitted head and helmet at this "
                "voxel resolution. Adjust the fit or gap and try again.";
        return false;
    }
    if (statistics.negative_samples == 0 || statistics.negative_samples == statistics.samples)
    {
        error = "The generated implicit solid does not intersect the sampling grid.";
        return false;
    }
    if (output.vertices.rows() == 0 || output.faces.rows() == 0 ||
        !output.vertices.allFinite())
    {
        error = "Marching tetrahedra did not produce a valid output mesh.";
        return false;
    }
    output.voxel_size = fine.spacing;

    std::size_t boundary_edges = 0;
    std::size_t nonmanifold_edges = 0;
    std::size_t boundary_grid_edges = 0;
    const Eigen::RowVector3d fine_last = fine.point(fine.nx - 1, fine.ny - 1, fine.nz - 1);
    if (!closed_two_manifold(
            output.faces, output.vertices, fine.minimum, fine_last, fine.spacing,
            boundary_edges, nonmanifold_edges, boundary_grid_edges))
    {
        error = "The fine output is not a closed two-manifold mesh; found " +
            std::to_string(boundary_edges) + " boundary edges and " +
            std::to_string(nonmanifold_edges) + " non-manifold edges (" +
            std::to_string(boundary_grid_edges) + " at grid bounds). A surface "
            "reaching a skipped block shows up here.";
        return false;
    }
    return true;
}

} // namespace

bool generate_final_helmet(
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    const std::vector<bool>& head_lid,
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::MatrixXi& helmet_faces,
    LatticeType lattice_type,
    GeneratedMesh& output,
    std::string& error,
    const GenerationOptions& options)
{
    const double lattice_room_target = options.lattice_room_target;
    const ShellMode shell_mode = options.shell_mode;
    const double wall_skin_cells = options.wall_skin_cells;
    const bool fine = options.fine_spacing > 0.0;
    if (!std::isfinite(options.fine_spacing) || options.fine_spacing < 0.0 ||
        options.fine_block_cells < 1 || options.fine_block_cells > 64)
    {
        error = "Invalid fine-grid options.";
        return false;
    }
    const bool reentrant = options.auxetic_cell == AuxeticCell::Reentrant;
    if (!std::isfinite(options.simplify_tolerance) || options.simplify_tolerance < 0.0 ||
        !std::isfinite(options.line_width) || options.line_width < 0.0)
    {
        error = "Invalid simplification tolerance or line width.";
        return false;
    }
    if (reentrant && lattice_type != LatticeType::Auxetic)
    {
        error = "The re-entrant cell is an auxetic lattice: use it with --lattice auxetic.";
        return false;
    }
    if (reentrant && !fine)
    {
        error = "The re-entrant cell's walls are thinner than the coarse grid can draw: "
                "give --fine-spacing as well.";
        return false;
    }
    if (head_vertices.rows() == 0 || head_faces.rows() == 0 ||
        helmet_vertices.rows() == 0 || helmet_faces.rows() == 0 ||
        head_vertices.cols() != 3 || helmet_vertices.cols() != 3 ||
        !head_vertices.allFinite() || !helmet_vertices.allFinite())
    {
        error = "Input head or helmet mesh is empty or invalid.";
        return false;
    }
    if (!head_lid.empty() &&
        head_lid.size() != static_cast<std::size_t>(head_faces.rows()))
    {
        error = "The head's lid list does not match its triangles.";
        return false;
    }
    if (shell_mode == ShellMode::Wall &&
        (!std::isfinite(wall_skin_cells) || wall_skin_cells < 0.0 || wall_skin_cells > 3.0))
    {
        error = "The wall skin must be between 0 and 3 grid cells.";
        return false;
    }

    const Eigen::RowVector3d head_min = head_vertices.colwise().minCoeff();
    const Eigen::RowVector3d head_max = head_vertices.colwise().maxCoeff();
    const Eigen::RowVector3d helmet_min = helmet_vertices.colwise().minCoeff();
    const Eigen::RowVector3d helmet_max = helmet_vertices.colwise().maxCoeff();
    const Eigen::RowVector3d combined_min = head_min.cwiseMin(helmet_min);
    const Eigen::RowVector3d combined_max = head_max.cwiseMax(helmet_max);
    const double head_height = head_max.y() - head_min.y();
    if (head_height <= 0.0 || !std::isfinite(head_height))
    {
        error = "The fitted head mesh has zero height.";
        return false;
    }

    double voxel_size = head_height / 120.0;
    Grid grid;
    auto set_grid_dimensions = [&]()
    {
        const double padding = voxel_size * 8.0;
        grid.minimum = combined_min.array() - padding;
        const Eigen::RowVector3d extent = combined_max - combined_min
            + Eigen::RowVector3d::Constant(padding * 2.0);
        grid.step = Eigen::RowVector3d::Constant(voxel_size);
        grid.nx = static_cast<unsigned>(std::ceil(extent.x() / voxel_size)) + 1;
        grid.ny = static_cast<unsigned>(std::ceil(extent.y() / voxel_size)) + 1;
        grid.nz = static_cast<unsigned>(std::ceil(extent.z() / voxel_size)) + 1;
    };
    set_grid_dimensions();
    // A loop, not a single correction: the grid has a fixed 8-voxel border on
    // each side that the cube-root estimate ignores, so one pass can undershoot
    // and the check below then fails (it did for any gap above ~10 mm).
    while (grid.sample_count() > max_grid_samples)
    {
        const double adjustment = std::cbrt(
            static_cast<double>(grid.sample_count()) / max_grid_samples);
        voxel_size *= adjustment * 1.01;
        set_grid_dimensions();
    }
    if (grid.sample_count() > max_grid_samples)
    {
        error = "The model bounds require a grid larger than the safe memory limit.";
        return false;
    }
    output.voxel_size = voxel_size;
    grid.minimum += voxel_size * Eigen::RowVector3d(0.173205, 0.223607, 0.316228);
    const Eigen::RowVector3d grid_last =
        grid.minimum + grid.step.cwiseProduct(Eigen::RowVector3d(
            grid.nx - 1, grid.ny - 1, grid.nz - 1));

    // The grid the surface is taken from: the coarse grid itself, or a fine
    // grid over it (coarse spacing / k, same origin) sampled only in blocks
    // the helmet layer passes through.
    FineGrid surface_grid;
    if (fine)
    {
        if (!make_fine_grid(grid.minimum, voxel_size, grid.nx, grid.ny, grid.nz,
                            options.fine_spacing, surface_grid, error))
        {
            return false;
        }
    }
    else
    {
        surface_grid.minimum = grid.minimum;
        surface_grid.spacing = voxel_size;
        surface_grid.nx = grid.nx;
        surface_grid.ny = grid.ny;
        surface_grid.nz = grid.nz;
        surface_grid.k = 1;
    }
    const double surface_spacing = surface_grid.spacing;

    // The re-entrant cell, checked before any heavy work.
    ReentrantCell cell;
    if (reentrant)
    {
        if (!make_reentrant_cell(options.cell_width, options.cell_angle_degrees,
                                 options.cell_h_ratio, options.wall_thickness, cell, error))
        {
            return false;
        }
        // Sizes are in the head file's units. A width outside 1-30% of the head
        // height means the units are not what the numbers assume (a head in mm
        // with defaults in metres, say).
        const double relative = cell.width / head_height;
        if (relative < 0.01 || relative > 0.30)
        {
            error = "The cell width " + std::to_string(cell.width) + " is " +
                fixed1(100.0 * relative) + "% of the head height; expected 1-30%. Check "
                "the units (sizes are in the head file's units, metres for our scan).";
            return false;
        }
        // The walls are vertical, so a tetrahedron edge spans at most sqrt 2
        // grid steps across one. Walls thinner than that can be stepped over
        // (holes); up to twice that they come out slightly thin in places.
        const double no_holes = cell.wall_thickness / std::sqrt(2.0);
        const double exact = cell.wall_thickness / (2.0 * std::sqrt(2.0));
        if (!(surface_spacing <= no_holes / 1.1))
        {
            error = "The surface grid spacing " + std::to_string(surface_spacing) +
                " is too coarse for walls " + std::to_string(cell.wall_thickness) +
                " thick (needs at most " + std::to_string(no_holes / 1.1) +
                ", or the walls get holes); use a smaller --fine-spacing.";
            return false;
        }
        if (options.simplify_tolerance > 0.25 * cell.wall_thickness)
        {
            error = "The simplification tolerance must be at most a quarter of the wall thickness.";
            return false;
        }
        if (surface_spacing > exact)
        {
            std::cout << "  NOTE: surface grid " << surface_spacing << " is coarser than "
                      << exact << " (wall / 2 sqrt 2): walls come out slightly thin in places.\n";
        }
        if (options.line_width > 0.0)
        {
            const double lines = cell.wall_thickness / options.line_width;
            if (std::abs(lines - std::round(lines)) > 0.05 || std::round(lines) < 1.0)
            {
                std::cout << "  WARNING: the wall (" << cell.wall_thickness << ") is "
                          << lines << " line widths (" << options.line_width
                          << "): not a whole number, so the slicer will under- or over-fill it.\n";
            }
        }
    }
    if (options.simplify_tolerance > surface_spacing)
    {
        error = "The simplification tolerance must not exceed the surface grid spacing.";
        return false;
    }

    std::cout << "Generating "
              << (lattice_type == LatticeType::Auxetic ? "auxetic" : "gyroid TPMS")
              << " lattice on a " << grid.nx << " x " << grid.ny << " x "
              << grid.nz << " grid (" << grid.sample_count() << " samples).\n";

    Eigen::MatrixXd points(grid.sample_count(), 3);
    for (unsigned z = 0; z < grid.nz; ++z)
    {
        for (unsigned y = 0; y < grid.ny; ++y)
        {
            for (unsigned x = 0; x < grid.nx; ++x)
            {
                points.row(grid.index(x, y, z)) = grid.point(x, y, z);
            }
        }
    }

    Eigen::VectorXd head_distance;
    Eigen::VectorXd helmet_distance;   // Band: unsigned; Wall: signed, < 0 inside the wall
    std::cout << "Computing head signed-distance field...\n";
    if (!compute_signed_distance(
            points, head_vertices, head_faces, head_distance, error, "head"))
    {
        return false;
    }
    if (shell_mode == ShellMode::Band)
    {
        std::cout << "Computing helmet distance field (unsigned)...\n";
        if (!compute_unsigned_distance(
                points, helmet_vertices, helmet_faces, helmet_distance,
                error, "helmet"))
        {
            return false;
        }
    }
    else
    {
        // The wall's inside comes from the winding number, which libigl takes
        // as |w| (so the triangles' winding direction does not matter) and
        // which tolerates the stock mesh's few open slits.
        std::cout << "Computing helmet wall signed-distance field...\n";
        if (!compute_signed_distance(
                points, helmet_vertices, helmet_faces, helmet_distance,
                error, "helmet wall"))
        {
            return false;
        }
        // Check the inside really is the wall: the samples inside it should add
        // up to the mesh's own enclosed volume. If the mesh is not a closed
        // solid, the winding number has no clean inside and this fails.
        double mesh_volume = 0.0;
        for (Eigen::Index f = 0; f < helmet_faces.rows(); ++f)
        {
            const Eigen::RowVector3d a = helmet_vertices.row(helmet_faces(f, 0));
            const Eigen::RowVector3d b = helmet_vertices.row(helmet_faces(f, 1));
            const Eigen::RowVector3d c = helmet_vertices.row(helmet_faces(f, 2));
            mesh_volume += a.dot(b.cross(c)) / 6.0;
        }
        mesh_volume = std::abs(mesh_volume);
        const double cell_volume = voxel_size * voxel_size * voxel_size;
        const double sampled_volume = cell_volume *
            static_cast<double>((helmet_distance.array() < 0.0).count());
        std::cout << "  helmet wall: " << sampled_volume << " (inside, sampled) vs "
                  << mesh_volume << " (mesh) model units cubed\n";
        if (!(mesh_volume > 0.0) ||
            std::abs(sampled_volume - mesh_volume) > 0.10 * mesh_volume)
        {
            error = "The helmet mesh is not a closed solid wall (sampled inside "
                    "volume differs from the mesh volume by more than 10%); use "
                    "--shell band.";
            return false;
        }
    }

    // "Inside the helmet" = inside the cavity mapped outward from the head's
    // centre. Previously it was "not inside the shell wall and inside the
    // helmet's bounding box", which filled the whole box with lattice.
    std::cout << "Mapping the helmet cavity from the head centre...\n";
    HelmetCavity cavity;
    if (!build_helmet_cavity(
            helmet_vertices, helmet_faces, (head_min + head_max) * 0.5,
            cavity, error))
    {
        return false;
    }
    std::cout << "  shell's inner face met first along "
              << fixed1(100.0 * cavity.inner_face_fraction) << "% of directions; "
              << cavity.vents_filled << " vent(s) bridged (largest "
              << fixed1(100.0 * cavity.largest_vent_fraction)
              << "% of directions); opening left open: "
              << fixed1(100.0 * cavity.open_fraction) << "%.\n";

    // A scanned head ends at a rim closed by a lid. Directions that see only
    // the lid get nothing, as the opening doesn't: liner and lattice stop at
    // the rim instead of wrapping the lid, which sealed the helmet shut.
    if (!head_lid.empty())
    {
        if (!exclude_unscanned_directions(
                cavity, head_vertices, head_faces, head_lid, error))
        {
            return false;
        }
        std::cout << "  head scanned up to its rim: "
                  << fixed1(100.0 * cavity.unscanned_fraction)
                  << "% of directions see only its lid, "
                  << fixed1(100.0 * cavity.unscanned_covered_fraction)
                  << "% of all directions also under the helmet: left empty "
                     "(head not measured there).\n";
    }

    const double head_clearance = voxel_size;
    const double shell_thickness = std::max(
        voxel_size * 2.25, head_height * 0.01);
    const double liner_thickness = std::max(
        voxel_size * 2.0, head_height * 0.005);
    const double liner_center_distance =
        head_clearance + liner_thickness * 0.5;
    // The shell as a field, negative inside, and how far it reaches in from
    // the stock wall's inner face (what it takes from the room for lattice).
    //   Band: within half a skin thickness of any face (the original).
    //   Wall: inside the wall, plus the skin on every face.
    // Skin in cells of the grid the surface is taken from: it only has to keep
    // walls thicker than that grid can resolve, so it shrinks with a finer grid.
    const double wall_skin = wall_skin_cells * surface_spacing;
    const double shell_inward =
        shell_mode == ShellMode::Band ? shell_thickness * 0.5 : wall_skin;
    std::vector<double> shell_field(grid.sample_count());
    for (std::size_t index = 0; index < grid.sample_count(); ++index)
    {
        shell_field[index] = shell_mode == ShellMode::Band
            ? std::abs(helmet_distance[index]) - shell_thickness * 0.5
            : helmet_distance[index] - wall_skin;
    }
    // Where the lattice stops, short of the shell's inner face along the ray.
    //   Band: half a skin short, where the band begins, so the two meet without
    //   a gap (the band is measured to the nearest shell point and the cut
    //   along the ray, and the along-ray distance is never the smaller).
    //   Wall: at the inner face itself, so the lattice overlaps the skin by
    //   wall_skin and fuses with it.
    // Under a bridged vent there is no wall; the lattice ends this far below
    // the bridged surface.
    const double cavity_margin =
        shell_mode == ShellMode::Band ? shell_thickness * 0.5 : 0.0;
    std::cout << "Shell: " << (shell_mode == ShellMode::Band
                                   ? "band (original: padded on both faces), "
                                   : "stock wall + skin ")
              << (shell_mode == ShellMode::Band ? shell_thickness : wall_skin)
              << (shell_mode == ShellMode::Band ? " thick" : " per face")
              << " (model units).\n";

    // How much room the fit leaves for lattice, for comparing fits (zoom vs
    // stretch). Report only: nothing below depends on it.
    double room_min = -1.0;
    {
        const double reserved = head_clearance + liner_thickness + shell_inward;
        LatticeRoom room;
        if (lattice_room(cavity, head_vertices, head_faces, reserved,
                         lattice_room_target, room, error))
        {
            room_min = room.min;
            std::cout << "Room for lattice (along rays from the head centre, over "
                         "the covered, scanned directions; head-to-shell distance "
                         "minus " << reserved << " for air, liner and the shell's inward part):\n"
                      << "  min " << room.min << ", p5 " << room.p5 << ", median "
                      << room.p50 << ", p95 " << room.p95 << ", max " << room.max
                      << "; medians top / middle / low: " << room.top << " / "
                      << room.middle << " / " << room.low << " (model units)";
            if (lattice_room_target > 0.0)
            {
                std::cout << "; under " << lattice_room_target << ": "
                          << fixed1(100.0 * room.under_target) << "% of directions";
            }
            std::cout << ".\n";
        }
        else
        {
            std::cout << "  (room report skipped: " << error << ")\n";
            error.clear();
        }
    }

    // Computed once and reused below. `covered`: the direction is one to build
    // in at all (helmet over it, head scanned), however far out the sample is.
    std::vector<double> inside_helmet(grid.sample_count());
    std::vector<char> covered(grid.sample_count());
    for (std::size_t index = 0; index < grid.sample_count(); ++index)
    {
        const Eigen::RowVector3d p = grid_point(grid, index);
        inside_helmet[index] = cavity.radial_distance(p, cavity_margin);
        covered[index] = cavity.covers(p) ? 1 : 0;
    }
    // Fine mode: covered directions grown by one coarse sample (26 neighbours),
    // so the coarse sweep and the block selection reach every fine sample in a
    // covered direction even where the covered region is narrower than a cell.
    std::vector<char> covered_near;
    if (fine)
    {
        covered_near.assign(grid.sample_count(), 0);
        for (unsigned z = 0; z < grid.nz; ++z)
        {
            for (unsigned y = 0; y < grid.ny; ++y)
            {
                for (unsigned x = 0; x < grid.nx; ++x)
                {
                    if (!covered[grid.index(x, y, z)])
                    {
                        continue;
                    }
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                const long nx = static_cast<long>(x) + dx;
                                const long ny = static_cast<long>(y) + dy;
                                const long nz = static_cast<long>(z) + dz;
                                if (nx >= 0 && ny >= 0 && nz >= 0 &&
                                    nx < static_cast<long>(grid.nx) &&
                                    ny < static_cast<long>(grid.ny) &&
                                    nz < static_cast<long>(grid.nz))
                                {
                                    covered_near[grid.index(
                                        static_cast<unsigned>(nx), static_cast<unsigned>(ny),
                                        static_cast<unsigned>(nz))] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Getting it on. The helmet must slide off the head (and, reversed, on)
    // without the head passing through any of it. Pick the direction the head
    // blocks least, judged where material can end up: the lattice space, the
    // liner (which can stand above the cavity under a vent) and the shell skin.
    std::vector<std::size_t> material_samples;
    std::vector<std::size_t> coarse;
    for (std::size_t index = 0; index < grid.sample_count(); ++index)
    {
        const bool lattice_space =
            inside_helmet[index] < 0.0 && head_distance[index] > head_clearance;
        const bool liner_space = covered[index] &&
            head_distance[index] > head_clearance &&
            head_distance[index] < liner_center_distance + liner_thickness * 0.5;
        const bool skin = shell_field[index] < 0.0;
        if (!lattice_space && !liner_space && !skin)
        {
            continue;
        }
        const std::size_t x = index % grid.nx;
        const std::size_t y = (index / grid.nx) % grid.ny;
        const std::size_t z = index / (static_cast<std::size_t>(grid.nx) * grid.ny);
        if (x % 4 == 0 && y % 4 == 0 && z % 4 == 0)
        {
            coarse.push_back(material_samples.size());
        }
        material_samples.push_back(index);
    }
    Eigen::MatrixXd material_points(
        static_cast<Eigen::Index>(material_samples.size()), 3);
    for (std::size_t k = 0; k < material_samples.size(); ++k)
    {
        material_points.row(static_cast<Eigen::Index>(k)) =
            points.row(static_cast<Eigen::Index>(material_samples[k]));
    }
    std::cout << "Choosing the direction the helmet lifts off...\n";
    LiftOff lift;
    if (!choose_lift_off(
            head_vertices, head_faces, material_points, coarse, lift, error))
    {
        return false;
    }
    const double material_count = static_cast<double>(material_samples.size());
    std::cout << "  lifts off " << describe_lift_off(lift) << "; on along the reverse.\n"
              << "  the head is in the way of "
              << fixed1(100.0 * lift.blocked / material_count)
              << "% of where material can go (lifting straight up: "
              << fixed1(100.0 * lift.blocked_straight_up / material_count)
              << "%); liner and lattice keep clear of it.\n";

    // The shell is the stock helmet and is not trimmed, so only report it.
    std::size_t skin_touching = 0;
    std::size_t skin_in_the_way = 0;
    for (std::size_t k = 0; k < material_samples.size(); ++k)
    {
        const std::size_t index = material_samples[k];
        if (shell_field[index] >= 0.0)
        {
            continue;
        }
        if (head_distance[index] < head_clearance)
        {
            ++skin_touching;
        }
        else if (lift.is_blocked[k])
        {
            ++skin_in_the_way;
        }
    }
    const double sample_volume = voxel_size * voxel_size * voxel_size;
    if (skin_touching > 0)
    {
        std::cout << "  WARNING: the helmet shell comes within the head clearance of "
                     "the head (" << skin_touching << " samples = "
                  << static_cast<double>(skin_touching) * sample_volume
                  << " model units cubed): the fit is tight there. Increase Gap.\n";
    }
    if (skin_in_the_way > 0)
    {
        std::cout << "  WARNING: the helmet shell itself is in the head's way when "
                     "lifting off (" << skin_in_the_way << " samples = "
                  << static_cast<double>(skin_in_the_way) * sample_volume
                  << " model units cubed). It is not trimmed.\n";
    }

    // The swept head: the space the head passes through as the helmet lifts
    // off. Liner and lattice are built against it instead of the head itself;
    // wherever nothing overhangs the two are the same. Computed throughout the
    // cavity, and in covered directions up to 7 voxels from the head (under a
    // vent the liner can stand above the cavity). Elsewhere it stays the head
    // distance, an upper bound: those samples lie in or beyond the shell wall,
    // where no liner belongs and the skin covers any overlap.
    std::vector<double> swept(
        head_distance.data(), head_distance.data() + head_distance.size());
    std::vector<char> swept_computed;   // 1 where the sweep was followed (accurate)
    {
        // Followed within the head's bounds widened by `margin`: further out
        // the head is at least that far, and liner and lattice only use values
        // up to liner_center_distance + liner_thickness / 2 (3 voxels).
        const double margin = 7.0 * voxel_size;
        const Eigen::RowVector3d box_min = head_min.array() - margin;
        const Eigen::RowVector3d box_max = head_max.array() + margin;
        if (liner_center_distance + liner_thickness * 0.5 + voxel_size > margin ||
            (box_min - grid.minimum).minCoeff() < 0.0 ||
            (grid_last - box_max).minCoeff() < 0.0)
        {
            error = "Internal: the swept-head margin does not fit the grid.";
            return false;
        }
        std::vector<std::size_t> near_cavity;
        for (std::size_t index = 0; index < grid.sample_count(); ++index)
        {
            if (inside_helmet[index] < 2.0 * voxel_size ||
                (covered[index] && head_distance[index] < margin) ||
                (fine && covered_near[index]))
            {
                near_cavity.push_back(index);
            }
        }
        GridField field;
        field.minimum = grid.minimum;
        field.spacing = voxel_size;
        field.nx = static_cast<int>(grid.nx);
        field.ny = static_cast<int>(grid.ny);
        field.nz = static_cast<int>(grid.nz);
        field.values = head_distance.data();
        swept_head_distance(
            field, lift.direction, box_min, box_max, margin, near_cavity, swept);
        swept_computed.assign(grid.sample_count(), 0);
        for (const std::size_t index : near_cavity)
        {
            swept_computed[index] = 1;
        }
    }

    double maximum_gap_half_width = 0.0;   // measured along rays: >= the true half-width
    std::size_t domain_samples = 0;
    std::size_t kept_clear = 0;
    for (std::size_t index = 0; index < grid.sample_count(); ++index)
    {
        if (inside_helmet[index] >= 0.0)
        {
            continue;
        }
        const double outside_head = -swept[index] + head_clearance;
        if (outside_head < 0.0)
        {
            ++domain_samples;
            maximum_gap_half_width = std::max(
                maximum_gap_half_width,
                std::min(-inside_helmet[index], -outside_head));
        }
        else if (head_distance[index] > head_clearance)
        {
            ++kept_clear;
        }
    }
    std::cout << "Space the lattice may fill: " << domain_samples << " samples = "
              << static_cast<double>(domain_samples) * sample_volume
              << " (model units cubed); kept clear for lifting off: " << kept_clear
              << " samples = " << static_cast<double>(kept_clear) * sample_volume
              << ".\n";
    if (maximum_gap_half_width < voxel_size * 1.5)
    {
        error = "The fitted head and helmet leave less than three voxels of usable "
                "gap; increase clearance or adjust model scale/position.";
        return false;
    }

    output.cell_size = std::min(head_height / 9.0, maximum_gap_half_width * 1.5);
    const Eigen::RowVector3d lattice_anchor = combined_min;
    std::cout << "Maximum sampled gap half-width: "
              << maximum_gap_half_width << " (model units).\n";
    if (reentrant)
    {
        output.cell_size = cell.width;
        double nu_x = 0.0;
        double nu_z = 0.0;
        reentrant_ideal_poisson(cell, nu_x, nu_z);
        std::cout << "Lattice: re-entrant honeycomb walls (vertical, extruded along Y), angle "
                  << cell.angle_degrees << " deg, h/l " << cell.h_ratio << ": l " << cell.l
                  << ", h " << cell.h << "; repeat " << cell.width << " across X x "
                  << 2.0 * cell.dz << " along Z (rows " << cell.dz << " apart); walls "
                  << cell.wall_thickness << ", clear gap " << cell.clear_gap
                  << " (model units; anchored to the combined model bounds).\n"
                  << "  ideal (thin-wall) Poisson's ratio: " << nu_x << " squeezed along X, "
                  << nu_z << " along Z; for these walls run analysis_tools/poisson_frame.py "
                     "--cell-width/--cell-angle/--wall-thickness (in mm).\n";
        if (room_min > 0.0)
        {
            std::cout << "  through the thinnest room (" << room_min << "): about "
                      << room_min / cell.dz << " rows along Z, " << room_min / cell.dx
                      << " columns along X.\n";
        }
    }
    else
    {
        std::cout << "Lattice unit-cell pitch: " << output.cell_size
                  << " (anchored to the combined model bounds).\n";
    }
    if (fine)
    {
        const FineContext context{
            grid, surface_grid, options, head_vertices, head_faces, helmet_vertices,
            helmet_faces, cavity, lift, head_distance, shell_field, swept, swept_computed,
            inside_helmet, covered_near, head_min, head_max, grid_last, lattice_type,
            reentrant ? &cell : nullptr,
            output.cell_size, lattice_anchor, shell_thickness, wall_skin,
            liner_center_distance, liner_thickness, head_clearance, cavity_margin};
        return generate_fine_surface(context, output, error);
    }

    std::vector<float> lattice_distance(
        grid.sample_count(), std::numeric_limits<float>::infinity());
    if (lattice_type == LatticeType::Auxetic)
    {
        // Padded by the radius only, exactly as the original rasterizer.
        const double radius = auxetic_strut_radius(output.cell_size);
        SampleBox box;
        box.origin = grid.minimum;
        box.spacing = voxel_size;
        box.count[0] = grid.nx;
        box.count[1] = grid.ny;
        box.count[2] = grid.nz;
        for (const Strut& strut :
             auxetic_struts(grid.minimum, grid_last, output.cell_size, lattice_anchor))
        {
            rasterize_strut<float>(box, strut, radius, radius, lattice_distance);
        }
    }
    else
    {
        for (unsigned z = 0; z < grid.nz; ++z)
        {
            for (unsigned y = 0; y < grid.ny; ++y)
            {
                for (unsigned x = 0; x < grid.nx; ++x)
                {
                    lattice_distance[grid.index(x, y, z)] = static_cast<float>(
                        gyroid_distance(grid.point(x, y, z), output.cell_size, lattice_anchor));
                }
            }
        }
    }

    Eigen::VectorXd final_distance(grid.sample_count());
    std::size_t lattice_samples = 0;
    for (std::size_t index = 0; index < grid.sample_count(); ++index)
    {
        const double shell = shell_field[index];
        // The liner follows the swept head, and only in directions the helmet
        // covers where the head was scanned: it used to wrap the whole head,
        // lid and all. Cut by direction only, not by distance: where it meets
        // the shell the skin covers it, and under a vent it must not be holed.
        const double inner_liner = covered[index]
            ? std::abs(swept[index] - liner_center_distance) - liner_thickness * 0.5
            : cavity.open_distance;
        const double outside_head = -swept[index] + head_clearance;
        const double clipped_lattice = std::max({
            static_cast<double>(lattice_distance[index]),
            inside_helmet[index],
            outside_head
        });
        if (clipped_lattice < 0.0)
        {
            ++lattice_samples;
        }
        final_distance[index] = std::min({shell, inner_liner, clipped_lattice});
    }

    std::cout << "Lattice material occupies " << lattice_samples
              << " sampled grid points.\n";
    if (lattice_samples == 0)
    {
        error = "No lattice fits between the fitted head and helmet at this "
                "voxel resolution. Adjust the fit or gap and try again.";
        return false;
    }

    if (!(final_distance.minCoeff() < 0.0 && final_distance.maxCoeff() > 0.0))
    {
        error = "The generated implicit solid does not intersect the sampling grid.";
        return false;
    }

    const double isovalue = voxel_size * 0.05;
    MeshStatistics statistics;
    if (!mesh_blocks(
            surface_grid, {whole_grid_block(surface_grid)},
            [&](const FineBlock&, std::vector<double>& values)
            {
                std::copy(final_distance.data(),
                          final_distance.data() + final_distance.size(), values.begin());
            },
            isovalue, output.vertices, output.faces, statistics, error,
            options.simplify_tolerance))
    {
        return false;
    }
    log_simplification(statistics, options.simplify_tolerance);
    if (output.vertices.rows() == 0 || output.faces.rows() == 0 ||
        !output.vertices.allFinite())
    {
        error = "Marching cubes did not produce a valid output mesh.";
        return false;
    }

    std::size_t boundary_edges = 0;
    std::size_t nonmanifold_edges = 0;
    std::size_t boundary_grid_edges = 0;
    const Eigen::RowVector3d grid_max =
        grid.minimum + grid.step.cwiseProduct(Eigen::RowVector3d(
            grid.nx - 1, grid.ny - 1, grid.nz - 1));
    if (!closed_two_manifold(
            output.faces, output.vertices, grid.minimum, grid_max, voxel_size,
            boundary_edges, nonmanifold_edges, boundary_grid_edges))
    {
        error = "The sampled output is not a closed two-manifold mesh; "
                "found " + std::to_string(boundary_edges) + " boundary edges and "
                + std::to_string(nonmanifold_edges) + " non-manifold edges ("
                + std::to_string(boundary_grid_edges) + " at grid bounds).";
        return false;
    }
    return true;
}
