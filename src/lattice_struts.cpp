#include "lattice_struts.h"

#include <algorithm>
#include <array>
#include <cmath>

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
