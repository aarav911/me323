#include "lattice_struts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{

constexpr double pi = 3.14159265358979323846;

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

double segment_distance_2d(
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& end)
{
    const Eigen::Vector2d edge = end - start;
    const double edge_squared_length = edge.squaredNorm();
    const double t = edge_squared_length > 0.0
        ? std::clamp((point - start).dot(edge) / edge_squared_length, 0.0, 1.0)
        : 0.0;
    return (point - (start + t * edge)).norm();
}

// Closest approach of two segments that do not cross (none of the cell's
// walls cross: they only meet at nodes).
double segment_segment_distance_2d(
    const std::array<Eigen::Vector2d, 2>& a,
    const std::array<Eigen::Vector2d, 2>& b)
{
    return std::min({segment_distance_2d(a[0], b[0], b[1]),
                     segment_distance_2d(a[1], b[0], b[1]),
                     segment_distance_2d(b[0], a[0], a[1]),
                     segment_distance_2d(b[1], a[0], a[1])});
}

// One node's three walls, with the integer node ids of their ends: (row,
// column, 0 = bottom / 1 = top).
struct NodeWalls
{
    std::array<std::array<Eigen::Vector2d, 2>, 3> walls;
    std::array<std::array<std::array<int, 3>, 2>, 3> ends;
};

NodeWalls node_walls(const ReentrantCell& cell, int r, int c)
{
    const Eigen::Vector2d bottom(c * cell.dx, r * cell.dz);
    const Eigen::Vector2d top(c * cell.dx, r * cell.dz + cell.h);
    NodeWalls node;
    node.walls[0] = {bottom, top};
    node.ends[0] = {{{r, c, 0}, {r, c, 1}}};
    for (int side = 0; side < 2; ++side)
    {
        const int next = side == 0 ? c - 1 : c + 1;
        node.walls[1 + side] = {top, Eigen::Vector2d(next * cell.dx, (r + 1) * cell.dz)};
        node.ends[1 + side] = {{{r, c, 1}, {r + 1, next, 0}}};
    }
    return node;
}

bool same_parity(int r, int c)
{
    return ((c - r) % 2) == 0;    // also right for negatives: -1 % 2 == -1 != 0
}

} // namespace

double auxetic_strut_radius(double cell_size)
{
    return cell_size * 0.12;
}

std::vector<Strut> auxetic_struts(
    const Eigen::RowVector3d& grid_min,
    const Eigen::RowVector3d& grid_max,
    double cell_size,
    const Eigen::RowVector3d& anchor)
{
    const double inward_fraction = 0.42;
    const int first_x = static_cast<int>(std::floor(
        (grid_min.x() - anchor.x()) / cell_size)) - 1;
    const int last_x = static_cast<int>(std::ceil(
        (grid_max.x() - anchor.x()) / cell_size)) + 1;
    const int first_z = static_cast<int>(std::floor(
        (grid_min.z() - anchor.z()) / cell_size)) - 1;
    const int last_z = static_cast<int>(std::ceil(
        (grid_max.z() - anchor.z()) / cell_size)) + 1;
    const int first_y = static_cast<int>(std::floor(
        (grid_min.y() - anchor.y()) / cell_size)) - 1;
    const int last_y = static_cast<int>(std::ceil(
        (grid_max.y() - anchor.y()) / cell_size)) + 1;

    std::vector<Strut> struts;
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
                    struts.push_back({lower[node], lower[next]});
                    struts.push_back({lower[node], upper[node]});
                }
            }
        }
    }
    return struts;
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

double gyroid_distance(
    const Eigen::RowVector3d& point,
    double cell_size,
    const Eigen::RowVector3d& anchor)
{
    const double wall_thickness = cell_size * 0.20;
    const double distance_scale = cell_size / (2.0 * pi * std::sqrt(3.0));
    const double px = 2.0 * pi * (point.x() - anchor.x()) / cell_size;
    const double py = 2.0 * pi * (point.y() - anchor.y()) / cell_size;
    const double pz = 2.0 * pi * (point.z() - anchor.z()) / cell_size;
    const double gyroid =
        std::sin(px) * std::cos(py)
        + std::sin(py) * std::cos(pz)
        + std::sin(pz) * std::cos(px);
    return std::abs(gyroid) * distance_scale - wall_thickness * 0.5;
}

double reentrant_ideal_h_ratio(double angle_degrees)
{
    const double angle = angle_degrees * pi / 180.0;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return c * c / (-s) - s;
}

bool make_reentrant_cell(
    double width, double angle_degrees, double h_ratio, double wall_thickness,
    ReentrantCell& cell, std::string& error)
{
    cell = ReentrantCell{};
    if (!std::isfinite(width) || !std::isfinite(angle_degrees) || !std::isfinite(h_ratio) ||
        !std::isfinite(wall_thickness) || !(width > 0.0) || !(wall_thickness > 0.0) ||
        h_ratio < 0.0)
    {
        error = "Re-entrant cell: width and wall thickness must be positive, h/l not negative.";
        return false;
    }
    if (angle_degrees < -60.0 || angle_degrees > -10.0)
    {
        error = "Re-entrant cell: the angle must be from -60 to -10 degrees (negative = re-entrant).";
        return false;
    }
    const double angle = angle_degrees * pi / 180.0;
    cell.width = width;
    cell.angle_degrees = angle_degrees;
    cell.h_ratio = h_ratio > 0.0 ? h_ratio : reentrant_ideal_h_ratio(angle_degrees);
    cell.wall_thickness = wall_thickness;
    cell.l = width / (2.0 * std::cos(angle));
    cell.h = cell.h_ratio * cell.l;
    cell.dx = cell.l * std::cos(angle);
    cell.slant_dz = cell.l * std::sin(angle);
    cell.dz = cell.h + cell.slant_dz;
    if (!(cell.dz > 1e-9 * width))
    {
        error = "Re-entrant cell: h/l too small, the rows do not advance (need h > -l sin(angle)).";
        return false;
    }
    // Closest approach of walls that share no node, over a patch big enough to
    // hold every kind of neighbour (the layout repeats every 2 rows and 2 columns).
    std::vector<std::array<Eigen::Vector2d, 2>> walls;
    std::vector<std::array<std::array<int, 3>, 2>> ends;
    for (int r = -4; r <= 4; ++r)
    {
        for (int c = -6; c <= 6; ++c)
        {
            if (!same_parity(r, c))
            {
                continue;
            }
            const NodeWalls node = node_walls(cell, r, c);
            for (int k = 0; k < 3; ++k)
            {
                walls.push_back(node.walls[k]);
                ends.push_back(node.ends[k]);
            }
        }
    }
    double gap = std::numeric_limits<double>::infinity();
    for (std::size_t a = 0; a < walls.size(); ++a)
    {
        for (std::size_t b = 0; b < a; ++b)
        {
            const bool touch = ends[a][0] == ends[b][0] || ends[a][0] == ends[b][1] ||
                               ends[a][1] == ends[b][0] || ends[a][1] == ends[b][1];
            if (!touch)
            {
                gap = std::min(gap, segment_segment_distance_2d(walls[a], walls[b]));
            }
        }
    }
    cell.centreline_gap = gap;
    cell.clear_gap = gap - wall_thickness;
    if (!(cell.clear_gap >= wall_thickness))
    {
        error = "Re-entrant cell: walls that do not meet come within " +
            std::to_string(cell.clear_gap) + " of each other (clear), less than the wall "
            "thickness " + std::to_string(wall_thickness) + "; use thinner walls, a larger "
            "h/l or a less steep angle.";
        return false;
    }
    return true;
}

void reentrant_ideal_poisson(const ReentrantCell& cell, double& nu_x, double& nu_z)
{
    const double angle = cell.angle_degrees * pi / 180.0;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    nu_x = c * c / ((cell.h_ratio + s) * s);
    nu_z = 1.0 / nu_x;
}

std::vector<std::array<Eigen::Vector2d, 2>> reentrant_segments(
    const ReentrantCell& cell, int r0, int r1, int c0, int c1)
{
    std::vector<std::array<Eigen::Vector2d, 2>> walls;
    for (int r = r0; r <= r1; ++r)
    {
        for (int c = c0; c <= c1; ++c)
        {
            if (same_parity(r, c))
            {
                const NodeWalls node = node_walls(cell, r, c);
                walls.insert(walls.end(), node.walls.begin(), node.walls.end());
            }
        }
    }
    return walls;
}

double reentrant_wall_distance(
    const Eigen::RowVector3d& p,
    const ReentrantCell& cell,
    const Eigen::RowVector3d& anchor,
    double reach)
{
    const Eigen::Vector2d q(p.x() - anchor.x(), p.z() - anchor.z());
    // A node's walls lie in x in [(c - 1) dx, (c + 1) dx] and z in
    // [r dz, r dz + h] (the slanted ones end at (r + 1) dz < r dz + h): take
    // every node whose walls' box comes within `reach` of q.
    const int r_low = static_cast<int>(std::ceil((q.y() - cell.h - reach) / cell.dz));
    const int r_high = static_cast<int>(std::floor((q.y() + reach) / cell.dz));
    const int c_low = static_cast<int>(std::ceil((q.x() - reach) / cell.dx - 1.0));
    const int c_high = static_cast<int>(std::floor((q.x() + reach) / cell.dx + 1.0));
    double best = reach;
    for (int r = r_low; r <= r_high; ++r)
    {
        for (int c = c_low; c <= c_high; ++c)
        {
            if (!same_parity(r, c))
            {
                continue;
            }
            const NodeWalls node = node_walls(cell, r, c);
            for (const auto& wall : node.walls)
            {
                best = std::min(best, segment_distance_2d(q, wall[0], wall[1]));
            }
        }
    }
    return best - 0.5 * cell.wall_thickness;
}

template <typename Value>
void rasterize_strut(
    const SampleBox& box,
    const Strut& strut,
    double radius,
    double pad,
    std::vector<Value>& values)
{
    const Eigen::Vector3d min = strut.a.cwiseMin(strut.b).array() - pad;
    const Eigen::Vector3d max = strut.a.cwiseMax(strut.b).array() + pad;
    long low[3];
    long high[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        low[axis] = static_cast<long>(std::floor(
            (min[axis] - box.origin[axis]) / box.spacing));
        high[axis] = static_cast<long>(std::ceil(
            (max[axis] - box.origin[axis]) / box.spacing));
        low[axis] = std::max(low[axis], box.first[axis]);
        high[axis] = std::min(high[axis], box.first[axis] + box.count[axis] - 1);
    }
    for (long z = low[2]; z <= high[2]; ++z)
    {
        for (long y = low[1]; y <= high[1]; ++y)
        {
            for (long x = low[0]; x <= high[0]; ++x)
            {
                const long i = x - box.first[0];
                const long j = y - box.first[1];
                const long k = z - box.first[2];
                const Eigen::Vector3d sample = box.point(i, j, k);
                const double distance =
                    point_segment_distance(sample, strut.a, strut.b) - radius;
                Value& current = values[static_cast<std::size_t>(i)
                    + static_cast<std::size_t>(box.count[0])
                        * (static_cast<std::size_t>(j)
                           + static_cast<std::size_t>(box.count[1]) * static_cast<std::size_t>(k))];
                current = std::min(current, static_cast<Value>(distance));
            }
        }
    }
}

template void rasterize_strut<float>(
    const SampleBox&, const Strut&, double, double, std::vector<float>&);
template void rasterize_strut<double>(
    const SampleBox&, const Strut&, double, double, std::vector<double>&);
