#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

enum class LatticeType
{
    Auxetic,
    Gyroid
};

// How the helmet shell is made solid.
//   Wall: the stock helmet's own wall (inside its closed mesh, by winding
//         number) plus a thin skin of wall_skin_cells grid cells on every
//         face, so walls thinner than the grid can resolve do not open up.
//   Band: the original: everything within half a skin thickness of ANY face,
//         inner and outer, which pads the wall on both sides (about twice the
//         stock wall's material) and leaves hollows where the wall is thicker
//         than the band. For helmet meshes that are not closed solids.
enum class ShellMode
{
    Wall,
    Band
};

struct GeneratedMesh
{
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    double voxel_size = 0.0;     // spacing the surface was taken at (fine or coarse)
    double cell_size = 0.0;
};

// head_lid: per head triangle, whether it is the lid closing a scan (see
// head_lid.h); empty for a complete head.
struct GenerationOptions
{
    double lattice_room_target = 0.0;    // for the room report (model units); 0 = none
    ShellMode shell_mode = ShellMode::Wall;
    double wall_skin_cells = 0.5;        // cells of the grid the surface is taken from
    // Fine narrow band (fine_grid.h): the final solid is sampled at this
    // spacing (model units), in blocks, only where the helmet layer is; the
    // fit, cavity, lift-off and cell size still come from the coarse grid.
    // 0 = coarse grid only (the original).
    double fine_spacing = 0.0;
    bool fine_exact = false;             // test: every fine distance exact, no bounds
    bool fine_all_blocks = false;        // test: evaluate every block, skip none
    unsigned fine_block_cells = 8;       // coarse cells per block side
};

bool generate_final_helmet(
    const Eigen::MatrixXd& head_vertices,
    const Eigen::MatrixXi& head_faces,
    const std::vector<bool>& head_lid,
    const Eigen::MatrixXd& helmet_vertices,
    const Eigen::MatrixXi& helmet_faces,
    LatticeType lattice_type,
    GeneratedMesh& output,
    std::string& error,
    const GenerationOptions& options = GenerationOptions{});
