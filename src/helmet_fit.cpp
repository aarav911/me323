#include "helmet_fit.h"

#include <igl/AABB.h>
#include <igl/Hit.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace
{

constexpr double pi = 3.14159265358979323846;

// Solid-angle-weighted quantile of `values` (sorted together with weights).
double weighted_quantile(const std::vector<std::pair<double, double>>& sorted,
                         double total_weight, double q)
{
    double cumulative = 0.0;
    for (const auto& [value, weight] : sorted)
    {
        cumulative += weight;
        if (cumulative >= q * total_weight)
        {
            return value;
        }
    }
    return sorted.back().first;
}

double median_of(std::vector<std::pair<double, double>> values)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    double total = 0.0;
    for (const auto& entry : values)
    {
        total += entry.second;
    }
    return weighted_quantile(values, total, 0.5);
}

} // namespace

Eigen::MatrixXd stretch_fit_helmet(
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::RowVector3d& helmet_box_center,
    const Eigen::Vector3d& stretch,
    double pitch_degrees,
    const Eigen::RowVector3d& head_center,
    const Eigen::Vector3d& offset)
{
    const double t = pitch_degrees * pi / 180.0;
    const double c = std::cos(t);
    const double s = std::sin(t);
    Eigen::MatrixXd fitted(helmet_vertices.rows(), 3);
    for (Eigen::Index v = 0; v < helmet_vertices.rows(); ++v)
    {
        const double x = (helmet_vertices(v, 0) - helmet_box_center.x()) * stretch.x();
        const double y = (helmet_vertices(v, 1) - helmet_box_center.y()) * stretch.y();
        const double z = (helmet_vertices(v, 2) - helmet_box_center.z()) * stretch.z();
        fitted(v, 0) = x + (head_center.x() + offset.x());
        fitted(v, 1) = (y * c + z * s) + (head_center.y() + offset.y());
        fitted(v, 2) = (-y * s + z * c) + (head_center.z() + offset.z());
    }
    return fitted;
}

bool lattice_room(
    const HelmetCavity& cavity,
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    double reserved,
    double target,
    LatticeRoom& room,
    std::string& error)
{
    room = LatticeRoom{};
    if (cavity.radius.empty() || head_vertices.rows() == 0 || head_faces.rows() == 0)
    {
        error = "Lattice room: no cavity map or head mesh.";
        return false;
    }
    // Rays against a copy scaled to a fixed size, as for the cavity map
    // (libigl's ray epsilon is absolute).
    const Eigen::RowVector3d vmin = head_vertices.colwise().minCoeff();
    const Eigen::RowVector3d vmax = head_vertices.colwise().maxCoeff();
    const double diagonal = (vmax - vmin).norm();
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
    {
        error = "Lattice room: the head has zero size.";
        return false;
    }
    const double scale = 1000.0 / diagonal;
    const Eigen::MatrixXd scaled = head_vertices * scale;
    const Eigen::RowVector3d origin = cavity.centre * scale;
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(scaled, head_faces);

    std::vector<std::pair<double, double>> all;
    std::vector<std::pair<double, double>> top, middle, low;
    double total = 0.0;
    double under = 0.0;
    for (int i = 0; i < cavity.n_theta; ++i)
    {
        const double theta = (i + 0.5) * cavity.step;
        const double weight = std::sin(theta);
        const double elevation = 90.0 - theta * 180.0 / pi;
        for (int j = 0; j < cavity.n_phi; ++j)
        {
            const double inner = cavity.radius[static_cast<std::size_t>(i) * cavity.n_phi + j];
            if (std::isnan(inner))
            {
                continue;
            }
            const double phi = (j + 0.5) * cavity.step;
            const Eigen::RowVector3d direction(
                std::sin(theta) * std::cos(phi), std::cos(theta),
                std::sin(theta) * std::sin(phi));
            igl::Hit<double> hit;
            if (!tree.intersect_ray(scaled, head_faces, origin, direction, hit))
            {
                continue;
            }
            const double value = inner - hit.t / scale - reserved;
            all.emplace_back(value, weight);
            total += weight;
            if (value < target)
            {
                under += weight;
            }
            (elevation > 50.0 ? top : elevation > 20.0 ? middle : low)
                .emplace_back(value, weight);
        }
    }
    if (all.empty())
    {
        error = "Lattice room: no direction is both covered and scanned.";
        return false;
    }
    std::sort(all.begin(), all.end());
    room.directions = all.size();
    room.min = all.front().first;
    room.max = all.back().first;
    room.p5 = weighted_quantile(all, total, 0.05);
    room.p50 = weighted_quantile(all, total, 0.50);
    room.p95 = weighted_quantile(all, total, 0.95);
    room.top = median_of(top);
    room.middle = median_of(middle);
    room.low = median_of(low);
    room.under_target = under / total;
    return true;
}
