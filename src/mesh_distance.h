#pragma once

#include <igl/AABB.h>
#include <igl/fast_winding_number.h>

#include <Eigen/Core>

#include <string>

// Distance to a triangle mesh at single points, with the search trees built
// once (igl::signed_distance() rebuilds them on every call, too slow to call
// per block of a fine grid). Same computations as igl::signed_distance():
//   signed:   unsigned distance from the AABB tree, sign 1 - 2|w| from the
//             fast winding number (igl::signed_distance_fast_winding_number);
//   unsigned: sqrt of the AABB tree's squared distance.
// The queries only read the trees, so they may run on many threads at once.
class MeshDistance
{
public:
    bool init(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
              bool is_signed, std::string& error);

    double operator()(const Eigen::RowVector3d& point) const;

    bool is_signed() const { return signed_; }

private:
    Eigen::MatrixXd vertices_;
    Eigen::MatrixXi faces_;
    igl::AABB<Eigen::MatrixXd, 3> tree_;
    igl::FastWindingNumberBVH winding_;
    bool signed_ = true;
    bool ready_ = false;
};
