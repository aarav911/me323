#include "print_export.h"

void make_print_copy(
    const Eigen::MatrixXd& model_vertices,
    double head_height,
    Eigen::MatrixXd& print_vertices,
    PrintPlacement& placement)
{
    placement = PrintPlacement{};
    placement.scale = head_height < 2.0 ? 1000.0 : 1.0;
    print_vertices.resize(model_vertices.rows(), 3);
    for (Eigen::Index v = 0; v < model_vertices.rows(); ++v)
    {
        // (x, y, z) -> (x, z, -y): a quarter turn (-90 degrees) about X, determinant +1.
        print_vertices(v, 0) = placement.scale * model_vertices(v, 0);
        print_vertices(v, 1) = placement.scale * model_vertices(v, 2);
        print_vertices(v, 2) = -placement.scale * model_vertices(v, 1);
    }
    if (print_vertices.rows() == 0)
    {
        return;
    }
    const Eigen::RowVector3d low = print_vertices.colwise().minCoeff();
    const Eigen::RowVector3d high = print_vertices.colwise().maxCoeff();
    const Eigen::RowVector3d shift(-0.5 * (low.x() + high.x()), -0.5 * (low.y() + high.y()), -low.z());
    print_vertices.rowwise() += shift;
    placement.size = high - low;
}
