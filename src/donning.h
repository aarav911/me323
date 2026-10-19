#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <string>
#include <vector>

// Getting the helmet on: it has to slide over the head without the head passing
// through any of it. The test here is a straight slide. If the helmet can lift
// off along one fixed direction, it goes on along the reverse. That is
// sufficient, not necessary (a helmet can also be rolled on), and it is checked
// against the head mesh only: the scanned surface plus its lid.
//
// Lifting straight up fails for the Run_2 scan. The back of the skull curves
// under, and about 10% of the lattice space lies beneath it. Tilting the
// lift-off ~25 degrees toward the back of the head clears nearly all of it.

// Up (+Y) tilted by `tilt_degrees` toward +Z (the face; negative = toward the
// back) and by `side_degrees` toward +X. Unit length.
Eigen::RowVector3d lift_off_direction(double tilt_degrees, double side_degrees);

struct LiftOff
{
    Eigen::RowVector3d direction = Eigen::RowVector3d(0.0, 1.0, 0.0);
    double tilt_degrees = 0.0;
    double side_degrees = 0.0;
    std::size_t blocked = 0;              // points the head is in the way of
    std::size_t blocked_straight_up = 0;  // the same, lifting straight up
    std::vector<char> is_blocked;         // per point, for `direction`
};

// Picks the lift-off direction that the head blocks least. A point is blocked
// when a ray from it along the direction meets the head: the head would pass
// through it as the helmet lifts. Candidates: tilts -45..45 and sideways
// -10..10 degrees, in 5-degree steps. All are ranked on the `coarse` subset of
// point indices; the near-best are re-counted on every point; the fewest
// blocked wins, and among equals the one nearest straight up.
bool choose_lift_off(
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    const Eigen::MatrixXd& points,
    const std::vector<std::size_t>& coarse,
    LiftOff& choice,
    std::string& error);

// A field sampled on the generator's grid: values[x + nx * (y + ny * z)] at
// minimum + spacing * (x, y, z).
struct GridField
{
    Eigen::RowVector3d minimum = Eigen::RowVector3d::Zero();
    double spacing = 0.0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    const double* values = nullptr;

    Eigen::RowVector3d point(std::size_t index) const;
    double trilinear(const Eigen::RowVector3d& p) const;   // clamps to the grid
};

// Distance to the space the head sweeps through as the helmet lifts off along
// `direction`: for each listed sample p, swept[p] = min over t >= 0 of
// head_distance(p + t * direction). That is the signed distance to the swept
// head (the distance to a union is the minimum of the distances). `swept`
// comes in holding head_distance itself, the t = 0 value.
//
// Samples are taken every half spacing along the ray, trilinearly. Since a
// distance changes by at most the distance moved, that overestimates the true
// value by at most a quarter spacing (plus interpolation error, negligible
// near the smooth head). The ray is only followed inside [box_min, box_max];
// with the box the head's bounds widened by `margin`, the result is exact in
// that sense wherever it is below `margin`, and is otherwise only known to be
// at least `margin`. A ray stops once the result drops below -margin.
void swept_head_distance(
    const GridField& head_distance,
    const Eigen::RowVector3d& direction,
    const Eigen::RowVector3d& box_min,
    const Eigen::RowVector3d& box_max,
    double margin,
    const std::vector<std::size_t>& samples,
    std::vector<double>& swept);

// The same, at one point p anywhere (not only at grid samples), stepping every
// `step` along the ray. `at_p` is the head distance at p itself (exact, from
// the mesh), which stands for the t = 0 term; the grid is only interpolated
// for t > 0. With p a grid sample, at_p its grid value and step half the
// spacing, this takes the same steps as swept_head_distance(); the two can
// differ only by rounding in that function's interpolated t = 0 sample.
double swept_head_distance_at(
    const GridField& head_distance,
    const Eigen::RowVector3d& direction,
    const Eigen::RowVector3d& box_min,
    const Eigen::RowVector3d& box_max,
    double margin,
    double step,
    const Eigen::RowVector3d& p,
    double at_p);
