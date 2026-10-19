#include "donning.h"

#include <igl/AABB.h>
#include <igl/Hit.h>
#include <igl/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

constexpr double pi = 3.14159265358979323846;

// Angle from straight up, for preferring the more vertical of equal choices.
double tilt_from_vertical(const Eigen::RowVector3d& direction)
{
    return std::acos(std::clamp(direction.y(), -1.0, 1.0));
}

} // namespace

Eigen::RowVector3d lift_off_direction(double tilt_degrees, double side_degrees)
{
    const double tilt = tilt_degrees * pi / 180.0;
    const double side = side_degrees * pi / 180.0;
    const Eigen::RowVector3d direction(
        std::sin(side) * std::cos(tilt), std::cos(side) * std::cos(tilt),
        std::sin(tilt));
    return direction.normalized();
}

bool choose_lift_off(
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    const Eigen::MatrixXd& points,
    const std::vector<std::size_t>& coarse,
    LiftOff& choice,
    std::string& error)
{
    choice = LiftOff{};
    if (head_vertices.rows() == 0 || head_faces.rows() == 0 ||
        head_vertices.cols() != 3 || head_faces.cols() != 3 ||
        !head_vertices.allFinite() || points.cols() != 3 || points.rows() == 0)
    {
        error = "Lift-off: empty or invalid head mesh or point list.";
        return false;
    }
    for (const std::size_t index : coarse)
    {
        if (index >= static_cast<std::size_t>(points.rows()))
        {
            error = "Lift-off: coarse subset indexes past the point list.";
            return false;
        }
    }

    // As for the cavity map: rays against a copy scaled to a fixed size, so
    // libigl's absolute ray epsilon can't hide the scan's small triangles.
    const Eigen::RowVector3d vmin = head_vertices.colwise().minCoeff();
    const Eigen::RowVector3d vmax = head_vertices.colwise().maxCoeff();
    const double diagonal = (vmax - vmin).norm();
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
    {
        error = "Lift-off: the head has zero size.";
        return false;
    }
    const double scale = 1000.0 / diagonal;
    const Eigen::MatrixXd scaled = head_vertices * scale;
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(scaled, head_faces);

    // Blocked flags for `count` points: all of them, or the coarse subset.
    const auto blocked_flags = [&](const Eigen::RowVector3d& direction, bool subset)
    {
        const std::size_t count = subset
            ? coarse.size() : static_cast<std::size_t>(points.rows());
        std::vector<char> flags(count, 0);
        igl::parallel_for(count, [&](const std::size_t k)
        {
            const Eigen::Index row = static_cast<Eigen::Index>(subset ? coarse[k] : k);
            const Eigen::RowVector3d origin = points.row(row) * scale;
            igl::Hit<double> hit;
            flags[k] = tree.intersect_ray(scaled, head_faces, origin, direction, hit)
                ? 1 : 0;
        }, 1000);
        return flags;
    };
    const auto total = [](const std::vector<char>& flags)
    {
        return static_cast<std::size_t>(std::count(flags.begin(), flags.end(), 1));
    };

    struct Candidate
    {
        double tilt;
        double side;
        Eigen::RowVector3d direction;
        std::size_t coarse_blocked;
    };
    std::vector<Candidate> candidates;
    for (int tilt = -45; tilt <= 45; tilt += 5)
    {
        for (int side = -10; side <= 10; side += 5)
        {
            const Eigen::RowVector3d direction = lift_off_direction(tilt, side);
            candidates.push_back({static_cast<double>(tilt), static_cast<double>(side),
                                  direction, coarse.empty()
                                      ? 0 : total(blocked_flags(direction, true))});
        }
    }
    // Stable, so equal candidates (e.g. 5 degrees either side) keep the order
    // they were made in, and the choice never depends on the sort's internals.
    std::stable_sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b)
              {
                  if (a.coarse_blocked != b.coarse_blocked)
                  {
                      return a.coarse_blocked < b.coarse_blocked;
                  }
                  return tilt_from_vertical(a.direction) < tilt_from_vertical(b.direction);
              });

    // A few coarse hits apart is sampling noise, so re-count everything within
    // that of the best on all points (at most six, nearest straight up first).
    const std::size_t tolerance = std::max<std::size_t>(
        3, static_cast<std::size_t>(0.0005 * static_cast<double>(coarse.size())));
    const std::size_t limit = candidates.front().coarse_blocked + tolerance;
    bool chosen = false;
    for (std::size_t c = 0; c < candidates.size() && c < 6; ++c)
    {
        if (candidates[c].coarse_blocked > limit)
        {
            break;
        }
        std::vector<char> flags = blocked_flags(candidates[c].direction, false);
        const std::size_t blocked = total(flags);
        const bool better = !chosen || blocked < choice.blocked ||
            (blocked == choice.blocked &&
             tilt_from_vertical(candidates[c].direction)
                 < tilt_from_vertical(choice.direction));
        if (better)
        {
            chosen = true;
            choice.direction = candidates[c].direction;
            choice.tilt_degrees = candidates[c].tilt;
            choice.side_degrees = candidates[c].side;
            choice.blocked = blocked;
            choice.is_blocked = std::move(flags);
        }
    }
    choice.blocked_straight_up =
        total(blocked_flags(Eigen::RowVector3d(0.0, 1.0, 0.0), false));
    return true;
}

Eigen::RowVector3d GridField::point(std::size_t index) const
{
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const std::size_t z = index / plane;
    const std::size_t y = (index % plane) / nx;
    const std::size_t x = index % nx;
    return minimum + spacing * Eigen::RowVector3d(
        static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
}

double GridField::trilinear(const Eigen::RowVector3d& p) const
{
    const int size[3] = {nx, ny, nz};
    int cell[3];
    double fraction[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        const double g = std::clamp(
            (p[axis] - minimum[axis]) / spacing, 0.0, size[axis] - 1.0);
        cell[axis] = std::min(static_cast<int>(std::floor(g)), size[axis] - 2);
        fraction[axis] = g - cell[axis];
    }
    const auto at = [&](int dx, int dy, int dz)
    {
        return values[static_cast<std::size_t>(cell[0] + dx)
            + static_cast<std::size_t>(nx)
                * (static_cast<std::size_t>(cell[1] + dy)
                   + static_cast<std::size_t>(ny) * (cell[2] + dz))];
    };
    double result = 0.0;
    for (int dz = 0; dz <= 1; ++dz)
    {
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const double w = (dx ? fraction[0] : 1.0 - fraction[0])
                    * (dy ? fraction[1] : 1.0 - fraction[1])
                    * (dz ? fraction[2] : 1.0 - fraction[2]);
                result += w * at(dx, dy, dz);
            }
        }
    }
    return result;
}

void swept_head_distance(
    const GridField& head_distance,
    const Eigen::RowVector3d& direction,
    const Eigen::RowVector3d& box_min,
    const Eigen::RowVector3d& box_max,
    double margin,
    const std::vector<std::size_t>& samples,
    std::vector<double>& swept)
{
    const double step = 0.5 * head_distance.spacing;
    igl::parallel_for(samples.size(), [&](const std::size_t k)
    {
        const std::size_t index = samples[k];
        double best = swept[index];
        if (best < -margin)
        {
            return;
        }
        const Eigen::RowVector3d p = head_distance.point(index);

        // The part of the ray p + t * direction, t >= 0, inside the box.
        double t_enter = 0.0;
        double t_leave = std::numeric_limits<double>::infinity();
        for (int axis = 0; axis < 3; ++axis)
        {
            if (direction[axis] == 0.0)
            {
                if (p[axis] < box_min[axis] || p[axis] > box_max[axis])
                {
                    return;
                }
                continue;
            }
            double a = (box_min[axis] - p[axis]) / direction[axis];
            double b = (box_max[axis] - p[axis]) / direction[axis];
            if (a > b)
            {
                std::swap(a, b);
            }
            t_enter = std::max(t_enter, a);
            t_leave = std::min(t_leave, b);
        }
        if (!(t_enter <= t_leave))
        {
            return;
        }
        for (std::size_t s = 0;; ++s)
        {
            const double t = std::min(t_enter + static_cast<double>(s) * step, t_leave);
            best = std::min(best, head_distance.trilinear(p + t * direction));
            if (best < -margin || t >= t_leave)
            {
                break;
            }
        }
        swept[index] = best;
    }, 1000);
}

double swept_head_distance_at(
    const GridField& head_distance,
    const Eigen::RowVector3d& direction,
    const Eigen::RowVector3d& box_min,
    const Eigen::RowVector3d& box_max,
    double margin,
    double step,
    const Eigen::RowVector3d& p,
    double at_p)
{
    double best = at_p;
    if (best < -margin)
    {
        return best;
    }
    // The part of the ray p + t * direction, t >= 0, inside the box.
    double t_enter = 0.0;
    double t_leave = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0.0)
        {
            if (p[axis] < box_min[axis] || p[axis] > box_max[axis])
            {
                return best;
            }
            continue;
        }
        double a = (box_min[axis] - p[axis]) / direction[axis];
        double b = (box_max[axis] - p[axis]) / direction[axis];
        if (a > b)
        {
            std::swap(a, b);
        }
        t_enter = std::max(t_enter, a);
        t_leave = std::min(t_leave, b);
    }
    if (!(t_enter <= t_leave))
    {
        return best;
    }
    // Same steps as swept_head_distance(); a t = 0 step is taken by at_p.
    for (std::size_t s = 0;; ++s)
    {
        const double t = std::min(t_enter + static_cast<double>(s) * step, t_leave);
        if (t > 0.0)
        {
            best = std::min(best, head_distance.trilinear(p + t * direction));
        }
        if (best < -margin || t >= t_leave)
        {
            break;
        }
    }
    return best;
}
