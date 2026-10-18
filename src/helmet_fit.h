#pragma once

#include "helmet_cavity.h"

#include <Eigen/Core>

#include <cstddef>
#include <string>

// The stretch fit (FitMode::Stretch in fit_config.h). The helmet's vertices
// are centred on helmet_box_center, multiplied by `stretch` along the helmet
// file's own axes, tilted about the side-to-side (X) axis by pitch_degrees
// (+ = the front, +Z, goes up), and moved so the box centre lands at
// head_center + offset. Stretching before tilting keeps the shape a stretched
// GrabCAD helmet (tilting first and stretching along the world axes would
// shear it). analysis_tools/stretch_fit_study.py uses the same steps.
Eigen::MatrixXd stretch_fit_helmet(
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::RowVector3d& helmet_box_center,
    const Eigen::Vector3d& stretch,
    double pitch_degrees,
    const Eigen::RowVector3d& head_center,
    const Eigen::Vector3d& offset);

// Room for lattice seen from the head centre: in every direction the helmet
// covers where the head was scanned (finite cavity radius; vents bridged),
// the distance from the head to the shell's inner face along the ray, minus
// `reserved` (air + liner + the inward half of the skin, as the generator
// builds them). Statistics are weighted by solid angle. Along a ray rather
// than along the head's normal, so it can read high where surfaces are
// oblique; tools/check_output.py --step3 measures along the normal.
struct LatticeRoom
{
    double min = 0.0;
    double p5 = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
    double top = 0.0;          // median above 50 degrees elevation
    double middle = 0.0;       // 20..50 degrees
    double low = 0.0;          // below 20 degrees
    double under_target = 0.0; // fraction of the measured directions
    std::size_t directions = 0;
};

bool lattice_room(
    const HelmetCavity& cavity,
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    double reserved,
    double target,
    LatticeRoom& room,
    std::string& error);
