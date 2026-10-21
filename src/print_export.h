#pragma once

#include <Eigen/Core>

// The helmet as a printer wants it (step 4d): slicers take millimetres with Z
// up, while the model is in the head file's units (metres for our scan) with
// Y up. The print copy is
//   - scaled to mm: x1000 if the head is under 2 units tall (metres), else
//     left as it is (taken to be mm already);
//   - turned crown-down by a proper rotation, (x, y, z) -> (x, z, -y): the
//     crown (largest Y) ends up lowest, the walls (along Y) run along the
//     printer's Z, and nothing is mirrored (a mirror would swap left and right
//     and turn every triangle inside out); the faces are unchanged;
//   - set on the bed (lowest point at z = 0) with its footprint centred.
struct PrintPlacement
{
    double scale = 1.0;                  // model units -> mm
    Eigen::RowVector3d size = Eigen::RowVector3d::Zero();   // printed W x D x H, mm
};

void make_print_copy(
    const Eigen::MatrixXd& model_vertices,
    double head_height,
    Eigen::MatrixXd& print_vertices,
    PrintPlacement& placement);
