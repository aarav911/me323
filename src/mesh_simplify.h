#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

// Mesh simplification (step 4b) by edge collapses, on libigl's decimation
// engine but with our own rules, because libigl's QSlim ones do not fit:
//   - error, guaranteed: every raw vertex and raw triangle centre stays within
//     `tolerance` of the simplified surface, and every simplified vertex lies
//     within `tolerance` of the raw surface (checked at each collapse);
//   - order and a fast first filter: each original face's plane counted once
//     (libigl weights by area, so its cost is not a distance); a collapse
//     must keep the new vertex within `tolerance` of every plane its two ends
//     have absorbed (sum of squared distances <= tolerance^2). On its own this
//     is not enough: near a sharp crease a point can be near both planes yet
//     far from the surface;
//   - placement: the quadric's minimiser nearest the edge's midpoint (libigl
//     inverts the quadric, which fails on flat areas, so flat walls would
//     never simplify), kept within half an edge length of the midpoint;
//   - no surrounding triangle may flip (turn by 60 degrees or more) or become
//     a needle (smallest angle under ~0.1 degree, which folds onto its
//     neighbours) or degenerate; libigl checks none of this;
//   - no new triangle may cross another triangle: where two surfaces come
//     within about twice the tolerance (the faces of a thin sliver) the
//     distance bound alone would allow it;
//   - locked vertices never move and are never removed (the fine mesher locks
//     every vertex shared with a neighbouring block, so both blocks still
//     agree; also exact-isovalue sample vertices, where sheets can touch);
//   - with a box given, no new vertex leaves it: a block's piece then stays
//     inside its block, so pieces simplified apart cannot cross each other
//     (they meet only on the shared face, where the vertices are locked).
// Open boundaries are closed with a point at infinity, as libigl's qslim()
// does, and that point and its faces are removed again afterwards.
enum class SimplifyOutcome
{
    Simplified,    // collapses done (possibly none were allowed)
    TooSmall,      // fewer than 4 faces: left as it was
    Refused        // not edge-manifold once closed (e.g. a pinched boundary): left as it was
};

struct SimplifyResult
{
    SimplifyOutcome outcome = SimplifyOutcome::TooSmall;
    std::size_t faces_before = 0;
    std::size_t faces_after = 0;
};

// Simplifies (V, F) into (U, G). `locked` has one flag per row of V. I maps
// each row of U to the row of V it came from: a locked vertex keeps its exact
// position (U.row(u) == V.row(I(u)), bit for bit). If the outcome is not
// Simplified, U = V, G = F, I = 0..n-1. Deterministic: the same input always
// gives the same output.
SimplifyResult simplify_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<char>& locked,
    double tolerance,
    Eigen::MatrixXd& U,
    Eigen::MatrixXi& G,
    Eigen::VectorXi& I,
    const Eigen::RowVector3d* box_min = nullptr,
    const Eigen::RowVector3d* box_max = nullptr);
