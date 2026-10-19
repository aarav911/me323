#include "mesh_distance.h"

#include <igl/signed_distance.h>

#include <cmath>

bool MeshDistance::init(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces,
                        bool is_signed, std::string& error)
{
    ready_ = false;
    if (vertices.rows() == 0 || faces.rows() == 0 || vertices.cols() != 3 ||
        faces.cols() != 3 || !vertices.allFinite())
    {
        error = "Mesh distance: empty or invalid mesh.";
        return false;
    }
    vertices_ = vertices;
    faces_ = faces;
    signed_ = is_signed;
    tree_.init(vertices_, faces_);
    if (signed_)
    {
        // As igl::signed_distance() builds it for SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER.
        igl::fast_winding_number(vertices_.template cast<float>().eval(), faces_, 2, winding_);
    }
    ready_ = true;
    return true;
}

double MeshDistance::operator()(const Eigen::RowVector3d& point) const
{
    if (signed_)
    {
        return igl::signed_distance_fast_winding_number(
            point, vertices_, faces_, tree_, winding_);
    }
    int face = -1;
    Eigen::RowVector3d closest;
    return std::sqrt(tree_.squared_distance(vertices_, faces_, point, face, closest));
}
