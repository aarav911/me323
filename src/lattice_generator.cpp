#include "lattice_generator.h"

#include "donning.h"
#include "helmet_cavity.h"
#include "helmet_fit.h"

#include <igl/signed_distance.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

constexpr std::size_t max_grid_samples = 4'000'000;
constexpr double pi = 3.14159265358979323846;

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

bool has_closed_two_manifold_edges(
    const Eigen::MatrixXi& faces,
    const Eigen::MatrixXd& vertices,
    const Eigen::RowVector3d& grid_min,
    const Eigen::RowVector3d& grid_max,
    double voxel_size,
    std::size_t& boundary_edges,
    std::size_t& nonmanifold_edges,
    std::size_t& boundary_grid_edges)
{
    std::unordered_map<std::uint64_t, unsigned int> edge_counts;
    edge_counts.reserve(static_cast<std::size_t>(faces.rows()) * 2);
    for (Eigen::Index face = 0; face < faces.rows(); ++face)
    {
        for (int edge = 0; edge < 3; ++edge)
        {
            const std::uint32_t a = static_cast<std::uint32_t>(faces(face, edge));
            const std::uint32_t b = static_cast<std::uint32_t>(faces(face, (edge + 1) % 3));
            const std::uint32_t low = std::min(a, b);
            const std::uint32_t high = std::max(a, b);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(low) << 32) | high;
            unsigned int& count = edge_counts[key];
            ++count;
        }
    }
    boundary_edges = 0;
    nonmanifold_edges = 0;
    boundary_grid_edges = 0;
    for (const auto& [key, count] : edge_counts)
    {
        if (count == 1)
        {
            ++boundary_edges;
            const auto a = static_cast<Eigen::Index>(key >> 32);
            const auto b = static_cast<Eigen::Index>(key & 0xffffffffu);
            for (const Eigen::Index vertex : {a, b})
            {
                const Eigen::RowVector3d position = vertices.row(vertex);
                if ((position - grid_min).minCoeff() < voxel_size * 1e-5 ||
                    (grid_max - position).minCoeff() < voxel_size * 1e-5)
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
    }
    return boundary_edges == 0 && nonmanifold_edges == 0;
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

void marching_tetrahedra(
    const Grid& grid,
    const Eigen::VectorXd& field,
    double isovalue,
    Eigen::MatrixXd& vertices_out,
    Eigen::MatrixXi& faces_out)
{
    constexpr int cube_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    constexpr int tetrahedra[6][4] = {
        {0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
        {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}
    };

    std::vector<Eigen::RowVector3d> vertices;
    std::vector<Eigen::Vector3i> faces;
    std::unordered_map<std::uint64_t, int> edge_vertices;
    edge_vertices.reserve(static_cast<std::size_t>(field.size()) / 2);

    const auto vertex_for_edge = [&](std::size_t a, std::size_t b) -> int
    {
        if (field[a] == isovalue)
        {
            const std::uint64_t key = (std::uint64_t{1} << 63) | a;
            const auto [it, inserted] =
                edge_vertices.emplace(key, static_cast<int>(vertices.size()));
            if (inserted)
            {
                vertices.emplace_back(grid_point(grid, a));
            }
            return it->second;
        }
        if (field[b] == isovalue)
        {
            const std::uint64_t key = (std::uint64_t{1} << 63) | b;
            const auto [it, inserted] =
                edge_vertices.emplace(key, static_cast<int>(vertices.size()));
            if (inserted)
            {
                vertices.emplace_back(grid_point(grid, b));
            }
            return it->second;
        }

        const std::size_t low = std::min(a, b);
        const std::size_t high = std::max(a, b);
        const std::uint64_t key =
            (static_cast<std::uint64_t>(low) << 32) | high;
        const auto [it, inserted] =
            edge_vertices.emplace(key, static_cast<int>(vertices.size()));
        if (inserted)
        {
            const double t = (isovalue - field[a]) / (field[b] - field[a]);
            vertices.emplace_back(
                grid_point(grid, a)
                + t * (grid_point(grid, b) - grid_point(grid, a)));
        }
        return it->second;
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
        orient_triangle(triangle, vertices, inside_center, outside_center);
        faces.push_back(triangle);
    };

    for (unsigned z = 0; z + 1 < grid.nz; ++z)
    {
        for (unsigned y = 0; y + 1 < grid.ny; ++y)
        {
            for (unsigned x = 0; x + 1 < grid.nx; ++x)
            {
                std::array<std::size_t, 8> cube_indices;
                for (int corner = 0; corner < 8; ++corner)
                {
                    const unsigned gx = x + cube_offsets[corner][0];
                    const unsigned gy = y + cube_offsets[corner][1];
                    const unsigned gz = z + cube_offsets[corner][2];
                    cube_indices[corner] = grid.index(gx, gy, gz);
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
                        const std::size_t index = cube_indices[local];
                        if (field[index] < isovalue)
                        {
                            inside[inside_count++] = local;
                            inside_center += grid_point(grid, index);
                        }
                        else
                        {
                            outside[outside_count++] = local;
                            outside_center += grid_point(grid, index);
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
                        const int a = vertex_for_edge(
                            cube_indices[inside[0]], cube_indices[outside[0]]);
                        const int b = vertex_for_edge(
                            cube_indices[inside[0]], cube_indices[outside[1]]);
                        const int c = vertex_for_edge(
                            cube_indices[inside[0]], cube_indices[outside[2]]);
                        append_oriented(a, b, c, inside_center, outside_center);
                    }
                    else if (inside_count == 3)
                    {
                        const int a = vertex_for_edge(
                            cube_indices[outside[0]], cube_indices[inside[0]]);
                        const int b = vertex_for_edge(
                            cube_indices[outside[0]], cube_indices[inside[1]]);
                        const int c = vertex_for_edge(
                            cube_indices[outside[0]], cube_indices[inside[2]]);
                        append_oriented(a, b, c, inside_center, outside_center);
                    }
                    else
                    {
                        const int a = vertex_for_edge(
                            cube_indices[inside[0]], cube_indices[outside[0]]);
                        const int b = vertex_for_edge(
                            cube_indices[inside[0]], cube_indices[outside[1]]);
                        const int c = vertex_for_edge(
                            cube_indices[inside[1]], cube_indices[outside[0]]);
                        const int d = vertex_for_edge(
                            cube_indices[inside[1]], cube_indices[outside[1]]);
                        append_oriented(a, b, c, inside_center, outside_center);
                        append_oriented(b, d, c, inside_center, outside_center);
                    }
                }
            }
        }
    }

    vertices_out.resize(static_cast<Eigen::Index>(vertices.size()), 3);
    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
        vertices_out.row(static_cast<Eigen::Index>(i)) = vertices[i];
    }
    faces_out.resize(static_cast<Eigen::Index>(faces.size()), 3);
    for (std::size_t i = 0; i < faces.size(); ++i)
    {
        faces_out.row(static_cast<Eigen::Index>(i)) = faces[i];
    }
}

double point_segment_distance(
    const Eigen::Vector3d& point,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end)
{
    const Eigen::Vector3d edge = end - start;
    const double edge_squared_length = edge.squaredNorm();
    const double t = edge_squared_length > 0.0
        ? std::clamp((point - start).dot(edge) / edge_squared_length, 0.0, 1.0)
        : 0.0;
    return (point - (start + t * edge)).norm();
}

void rasterize_strut(
    const Grid& grid,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    double radius,
    std::vector<float>& distances)
{
    Eigen::Vector3d min = start.cwiseMin(end).array() - radius;
    Eigen::Vector3d max = start.cwiseMax(end).array() + radius;
    Eigen::Vector3i low;
    Eigen::Vector3i high;
    for (int axis = 0; axis < 3; ++axis)
    {
        low[axis] = static_cast<int>(std::floor(
            (min[axis] - grid.minimum[axis]) / grid.step[axis]));
        high[axis] = static_cast<int>(std::ceil(
            (max[axis] - grid.minimum[axis]) / grid.step[axis]));
    }
    low = low.cwiseMax(Eigen::Vector3i::Zero());
    high = high.cwiseMin(Eigen::Vector3i(
        static_cast<int>(grid.nx) - 1,
        static_cast<int>(grid.ny) - 1,
        static_cast<int>(grid.nz) - 1));

    for (int z = low.z(); z <= high.z(); ++z)
    {
        for (int y = low.y(); y <= high.y(); ++y)
        {
            for (int x = low.x(); x <= high.x(); ++x)
            {
                const Eigen::Vector3d sample = grid.point(x, y, z);
                const double distance =
                    point_segment_distance(sample, start, end) - radius;
                float& current = distances[grid.index(x, y, z)];
                current = std::min(current, static_cast<float>(distance));
            }
        }
    }
}

std::array<Eigen::Vector3d, 6> reentrant_cell(
    double center_x,
    double center_z,
    double cell_size,
    double inward_fraction)
{
    const double half_x = cell_size * 0.5;
    const double half_z = cell_size * 0.5;
    const double inward = half_x * inward_fraction;
    return {{
        {center_x - half_x, 0.0, center_z - half_z},
        {center_x - inward, 0.0, center_z},
        {center_x - half_x, 0.0, center_z + half_z},
        {center_x + half_x, 0.0, center_z + half_z},
        {center_x + inward, 0.0, center_z},
        {center_x + half_x, 0.0, center_z - half_z}
    }};
}

void build_auxetic_field(
    const Grid& grid,
    double cell_size,
    const Eigen::RowVector3d& anchor,
    std::vector<float>& field)
{
    const double radius = cell_size * 0.12;
    const double inward_fraction = 0.42;
    const Eigen::RowVector3d maximum =
        grid.minimum + grid.step.cwiseProduct(Eigen::RowVector3d(
            grid.nx - 1, grid.ny - 1, grid.nz - 1));

    const int first_x = static_cast<int>(std::floor(
        (grid.minimum.x() - anchor.x()) / cell_size)) - 1;
    const int last_x = static_cast<int>(std::ceil(
        (maximum.x() - anchor.x()) / cell_size)) + 1;
    const int first_z = static_cast<int>(std::floor(
        (grid.minimum.z() - anchor.z()) / cell_size)) - 1;
    const int last_z = static_cast<int>(std::ceil(
        (maximum.z() - anchor.z()) / cell_size)) + 1;
    const int first_y = static_cast<int>(std::floor(
        (grid.minimum.y() - anchor.y()) / cell_size)) - 1;
    const int last_y = static_cast<int>(std::ceil(
        (maximum.y() - anchor.y()) / cell_size)) + 1;

    for (int ix = first_x; ix <= last_x; ++ix)
    {
        for (int iz = first_z; iz <= last_z; ++iz)
        {
            const auto base = reentrant_cell(
                anchor.x() + ix * cell_size,
                anchor.z() + iz * cell_size,
                cell_size, inward_fraction);
            for (int iy = first_y; iy <= last_y; ++iy)
            {
                std::array<Eigen::Vector3d, 6> lower;
                std::array<Eigen::Vector3d, 6> upper;
                for (std::size_t node = 0; node < base.size(); ++node)
                {
                    lower[node] = base[node];
                    lower[node].y() = anchor.y() + iy * cell_size;
                    upper[node] = base[node];
                    upper[node].y() = anchor.y() + (iy + 1) * cell_size;
                }

                for (std::size_t node = 0; node < base.size(); ++node)
                {
                    const std::size_t next = (node + 1) % base.size();
                    rasterize_strut(grid, lower[node], lower[next], radius, field);
                    rasterize_strut(grid, lower[node], upper[node], radius, field);
                }
            }
        }
    }
}

void build_gyroid_field(
    const Grid& grid,
    double cell_size,
    const Eigen::RowVector3d& anchor,
    std::vector<float>& field)
{
    const double wall_thickness = cell_size * 0.20;
    const double distance_scale = cell_size / (2.0 * pi * std::sqrt(3.0));
    for (unsigned z = 0; z < grid.nz; ++z)
    {
        for (unsigned y = 0; y < grid.ny; ++y)
        {
            for (unsigned x = 0; x < grid.nx; ++x)
            {
                const Eigen::RowVector3d point = grid.point(x, y, z);
                const double px = 2.0 * pi * (point.x() - anchor.x()) / cell_size;
                const double py = 2.0 * pi * (point.y() - anchor.y()) / cell_size;
                const double pz = 2.0 * pi * (point.z() - anchor.z()) / cell_size;
                const double gyroid =
                    std::sin(px) * std::cos(py)
                    + std::sin(py) * std::cos(pz)
                    + std::sin(pz) * std::cos(px);
                field[grid.index(x, y, z)] = static_cast<float>(
                    std::abs(gyroid) * distance_scale - wall_thickness * 0.5);
            }
        }
    }
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
    double lattice_room_target,
    ShellMode shell_mode,
    double wall_skin_cells)
{
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
    const double wall_skin = wall_skin_cells * voxel_size;
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
    {
        const double reserved = head_clearance + liner_thickness + shell_inward;
        LatticeRoom room;
        if (lattice_room(cavity, head_vertices, head_faces, reserved,
                         lattice_room_target, room, error))
        {
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
    {
        // Followed within the head's bounds widened by `margin`: further out
        // the head is at least that far, and liner and lattice only use values
        // up to liner_center_distance + liner_thickness / 2 (3 voxels).
        const double margin = 7.0 * voxel_size;
        const Eigen::RowVector3d grid_last =
            grid.minimum + grid.step.cwiseProduct(Eigen::RowVector3d(
                grid.nx - 1, grid.ny - 1, grid.nz - 1));
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
                (covered[index] && head_distance[index] < margin))
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
              << maximum_gap_half_width << " (model units).\n"
              << "Lattice unit-cell pitch: " << output.cell_size
              << " (anchored to the combined model bounds).\n";
    std::vector<float> lattice_distance(
        grid.sample_count(), std::numeric_limits<float>::infinity());
    if (lattice_type == LatticeType::Auxetic)
    {
        build_auxetic_field(
            grid, output.cell_size, lattice_anchor, lattice_distance);
    }
    else
    {
        build_gyroid_field(
            grid, output.cell_size, lattice_anchor, lattice_distance);
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
    marching_tetrahedra(
        grid, final_distance, isovalue, output.vertices, output.faces);
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
    if (!has_closed_two_manifold_edges(
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
