#include "coupon.h"

#include <algorithm>
#include <cmath>
#include <vector>

bool generate_coupon(
    const ReentrantCell& cell,
    const CouponSpec& spec,
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces,
    CouponReport& report,
    std::string& error)
{
    report = CouponReport{};
    const double s = spec.spacing;
    const double t = cell.wall_thickness;
    const bool chamfer = spec.roof_chamfer_degrees > 0.0;
    if (!(cell.width > 0.0) || !(t > 0.0))
    {
        error = "Coupon: no valid re-entrant cell.";
        return false;
    }
    if (!(s > 0.0) || !std::isfinite(s) || !(s <= t / std::sqrt(2.0) / 1.1))
    {
        error = "Coupon: the grid spacing must be positive and at most wall / (1.1 sqrt 2) "
                "(else the walls get holes).";
        return false;
    }
    if (!(spec.width > 2.0 * spec.side + cell.width) || !(spec.depth > 2.0 * spec.side + cell.width) ||
        !(spec.walls_height > 0.0) || !(spec.side >= 2.0 * s) ||
        !(spec.floor >= 2.0 * s) || !(spec.roof >= 2.0 * s))
    {
        error = "Coupon: it must hold at least a cell inside its sides, and floor, roof and "
                "sides must be at least two grid steps thick.";
        return false;
    }
    if (!(spec.simplify_tolerance >= 0.0) || spec.simplify_tolerance > 0.25 * t ||
        spec.simplify_tolerance > s)
    {
        error = "Coupon: the simplification tolerance must be at most a quarter wall and a grid step.";
        return false;
    }
    if (chamfer && (spec.roof_chamfer_degrees < 45.0 || spec.roof_chamfer_degrees > 80.0))
    {
        error = "Coupon: the roof chamfer must be 0 (none) or 45-80 degrees from horizontal.";
        return false;
    }

    const double height = spec.floor + spec.walls_height + spec.roof;
    report.height = height;
    // A full chamfer: each cell's roof rests on slopes from all its walls, so
    // they must reach the cell's largest opening.
    const double reach = chamfer ? 1.02 * cell.largest_opening : 0.0;
    report.chamfer_reach = reach;
    const double slope = spec.roof_chamfer_degrees * 3.14159265358979323846 / 180.0;
    if (chamfer && !(reach * std::tan(slope) < spec.walls_height - 2.0 * s))
    {
        error = "Coupon: the chamfer would reach down to the floor; use taller walls or a lower angle.";
        return false;
    }
    // Wall distances must be exact as far as the chamfer reaches (and a few
    // grid steps past a wall's surface either way).
    const double wall_reach = 0.5 * t + reach + 3.0 * s;
    const Eigen::RowVector3d anchor(0.5 * spec.width, 0.0, 0.5 * spec.depth);   // a column of upright walls through the middle

    const auto field = [&](const Eigen::RowVector3d& p)
    {
        const double box = std::max({std::abs(p.x() - 0.5 * spec.width) - 0.5 * spec.width,
                                     std::abs(p.y() - 0.5 * spec.depth) - 0.5 * spec.depth,
                                     std::abs(p.z() - 0.5 * height) - 0.5 * height});
        const double floor = p.z() - spec.floor;
        const double roof = (height - spec.roof) - p.z();
        const double side = std::min({p.x(), spec.width - p.x(), p.y(), spec.depth - p.y()}) - spec.side;
        // The pattern lies in the printer's XY plane, as in the crown-down
        // helmet: the wall function takes (model X, model Z) = (printer X, printer Y).
        const double walls = reentrant_wall_distance(
            Eigen::RowVector3d(p.x(), 0.0, p.y()), cell, anchor, wall_reach);
        double solid = std::min({floor, roof, side, walls});
        if (chamfer)
        {
            // The cells along the sides are bounded by the side walls too.
            solid = std::min(solid, roof_chamfer(std::min(walls, side), roof,
                                                 spec.roof_chamfer_degrees, reach));
        }
        return std::max(box, solid);
    };

    // The grid: two coarse cells of padding all round (so the border is
    // outside), an odd offset (so no sample sits on the coupon's planes), and
    // a virtual coarse spacing of 5 grid steps, as in the helmet.
    const double coarse = 5.0 * s;
    const Eigen::RowVector3d minimum =
        Eigen::RowVector3d::Constant(-2.0 * coarse) + coarse * Eigen::RowVector3d(0.173205, 0.223607, 0.316228);
    const unsigned cnx = static_cast<unsigned>(std::ceil((spec.width - minimum.x()) / coarse)) + 3;
    const unsigned cny = static_cast<unsigned>(std::ceil((spec.depth - minimum.y()) / coarse)) + 3;
    const unsigned cnz = static_cast<unsigned>(std::ceil((height - minimum.z()) / coarse)) + 3;
    FineGrid grid;
    if (!make_fine_grid(minimum, coarse, cnx, cny, cnz, s, grid, error))
    {
        return false;
    }
    std::vector<char> active(static_cast<std::size_t>(cnx - 1) * (cny - 1) * (cnz - 1), 1);
    std::size_t total = 0;
    const std::vector<FineBlock> blocks = active_blocks(grid, cnx, cny, cnz, active, 8, total);
    const BlockField block_field = [&](const FineBlock& b, std::vector<double>& values)
    {
        for (unsigned k = 0; k < b.sz; ++k)
            for (unsigned j = 0; j < b.sy; ++j)
                for (unsigned i = 0; i < b.sx; ++i)
                    values[b.local(i, j, k)] = field(grid.point(b.x0 + i, b.y0 + j, b.z0 + k));
    };
    if (!mesh_blocks(grid, blocks, block_field, 0.05 * s, vertices, faces, report.meshing, error,
                     spec.simplify_tolerance))
    {
        return false;
    }
    std::size_t boundary = 0, nonmanifold = 0, at_bounds = 0;
    const Eigen::RowVector3d last = grid.point(grid.nx - 1, grid.ny - 1, grid.nz - 1);
    if (faces.rows() == 0 ||
        !closed_two_manifold(faces, vertices, grid.minimum, last, grid.spacing, boundary, nonmanifold, at_bounds))
    {
        error = "Coupon: the mesh is not closed (" + std::to_string(boundary) + " boundary, " +
                std::to_string(nonmanifold) + " non-manifold edges).";
        return false;
    }
    remove_fragments(vertices, faces, 1e-4, report.fragments);
    // On the bed exactly (the surface comes out a fraction of a grid step
    // below z = 0, where the samples straddle the floor's underside).
    vertices.col(2).array() -= vertices.col(2).minCoeff();
    return true;
}
