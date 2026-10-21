#pragma once

#include "fine_grid.h"
#include "lattice_struts.h"
#include "mesh_components.h"

#include <Eigen/Core>

#include <string>

// A small print test (steps 4c/4d): the helmet's honeycomb walls between a
// floor (like the shell, on the bed) and a roof (like the liner), closed by
// side walls, so every cell is sealed as in the helmet. Built directly in the
// print frame: millimetres, Z up, on the bed (lowest point at z = 0), walls
// along Z, the pattern in XY with the same orientation as in the crown-down
// helmet file (printer X = model X, printer Y = model Z), centred. Printed
// crown-down, the helmet's liner is a
// roof over open cells, which TPU may not bridge; the roof here tests that.
// With a chamfer (roof_chamfer_degrees > 0) the roof rests on slopes that need
// no bridging; printing both answers whether the helmet needs them.
struct CouponSpec
{
    double width = 40.0;              // X
    double depth = 40.0;              // Y
    double floor = 4.0;               // the helmet's shell is ~4.3 mm
    double walls_height = 25.0;       // the helmet's lattice is 21.5-35 mm thick
    double roof = 3.7;                // the helmet's liner (2 coarse grid cells)
    double side = 1.2;                // side walls, as thick as the honeycomb walls
    double spacing = 0.4;             // surface grid, mm
    double simplify_tolerance = 0.05; // mm (printable mode)
    double roof_chamfer_degrees = 0.0;   // 0 = none; else 45-80 from horizontal
};

struct CouponReport
{
    double height = 0.0;              // floor + walls + roof
    double chamfer_reach = 0.0;       // horizontal reach from the walls (0: no chamfer)
    MeshStatistics meshing;
    FragmentReport fragments;
};

// `cell` in millimetres. Fails (with a message) on parameters that cannot
// give a sound coupon: grid too coarse for the walls, floor or roof thinner
// than two grid steps, a chamfer reaching the floor, and so on.
bool generate_coupon(
    const ReentrantCell& cell,
    const CouponSpec& spec,
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces,
    CouponReport& report,
    std::string& error);
