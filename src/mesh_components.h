#pragma once

#include <Eigen/Core>

#include <cstddef>

// The separate pieces of a closed mesh (step 4e). Pieces are joined through
// shared edges; in a closed edge-manifold mesh every such piece is itself
// closed, so its signed volume is meaningful: positive for a solid piece
// (faces outward), negative for an air pocket inside the solid (faces facing
// into it). The main body is the solid piece of largest volume.
struct FragmentReport
{
    std::size_t pieces = 0;
    std::size_t pockets = 0;              // negative volume: kept
    double pocket_volume = 0.0;           // as a positive number
    double main_volume = 0.0;
    std::size_t removed = 0;              // solid pieces below the threshold
    double removed_volume = 0.0;
    std::size_t large_extra = 0;          // solid pieces at or above it, besides the main body: kept
    double large_extra_volume = 0.0;
};

// Removes solid pieces smaller than relative_threshold x the main body's
// volume (loose fragments, which a printer cannot place), keeps the main
// body, every air pocket, and any larger extra solid piece (reported, as it
// would mean something came apart: never dropped silently). Unreferenced
// vertices are dropped. Deterministic. The mesh must be closed.
void remove_fragments(
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces,
    double relative_threshold,
    FragmentReport& report);
