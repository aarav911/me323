#include "mesh_simplify.h"

#include <igl/AABB.h>
#include <igl/circulation.h>
#include <igl/connect_boundary_to_infinity.h>
#include <igl/decimate.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/min_heap.h>
#include <igl/remove_unreferenced.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace
{

constexpr double infinity = std::numeric_limits<double>::infinity();

// Sum of squared distances to a set of planes: p'Ap + 2 b'p + c.
struct Quadric
{
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    double c = 0.0;

    Quadric operator+(const Quadric& other) const
    {
        Quadric sum;
        sum.A = A + other.A;
        sum.b = b + other.b;
        sum.c = c + other.c;
        return sum;
    }
    double operator()(const Eigen::Vector3d& p) const
    {
        return p.dot(A * p) + 2.0 * b.dot(p) + c;
    }
};

void leave_unchanged(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    Eigen::MatrixXd& U, Eigen::MatrixXi& G, Eigen::VectorXi& I)
{
    U = V;
    G = F;
    I = Eigen::VectorXi::LinSpaced(V.rows(), 0, static_cast<int>(V.rows()) - 1);
}

// Unit normal of a triangle times twice its area.
Eigen::Vector3d area_normal(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                            const Eigen::Vector3d& c)
{
    return (b - a).cross(c - a);
}

// Shape quality: twice the area over the sum of the squared edge lengths.
// About 0.29 for an equilateral triangle, about half the smallest angle (in
// radians) for a needle, 0 for a degenerate one. Scale-free.
double quality(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c)
{
    const double scale = (b - a).squaredNorm() + (c - a).squaredNorm() + (c - b).squaredNorm();
    return scale > 0.0 ? area_normal(a, b, c).norm() / scale : 0.0;
}

// No collapse may make a triangle worse than this (smallest angle ~0.1 deg):
// such needles fold onto their neighbours (a self-intersection).
constexpr double minimum_quality = 1e-3;

// Squared distance from p to the triangle abc (closest point by regions,
// Ericson, Real-Time Collision Detection 5.1.5).
double point_triangle_squared_distance(const Eigen::Vector3d& p, const Eigen::Vector3d& a,
                                       const Eigen::Vector3d& b, const Eigen::Vector3d& c)
{
    const Eigen::Vector3d ab = b - a, ac = c - a, ap = p - a;
    const double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) return ap.squaredNorm();
    const Eigen::Vector3d bp = p - b;
    const double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) return bp.squaredNorm();
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double v = d1 / (d1 - d3);
        return (p - (a + v * ab)).squaredNorm();
    }
    const Eigen::Vector3d cp = p - c;
    const double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return cp.squaredNorm();
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double w = d2 / (d2 - d6);
        return (p - (a + w * ac)).squaredNorm();
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + w * (c - b))).squaredNorm();
    }
    const double denominator = va + vb + vc;
    if (!(std::abs(denominator) > 0.0))
    {
        // Degenerate triangle: the nearest of its three edges.
        const auto segment = [&](const Eigen::Vector3d& s0, const Eigen::Vector3d& s1)
        {
            const Eigen::Vector3d e = s1 - s0;
            const double l2 = e.squaredNorm();
            const double t = l2 > 0.0 ? std::clamp((p - s0).dot(e) / l2, 0.0, 1.0) : 0.0;
            return (p - (s0 + t * e)).squaredNorm();
        };
        return std::min({segment(a, b), segment(b, c), segment(c, a)});
    }
    const double v = vb / denominator, w = vc / denominator;
    return (p - (a + ab * v + ac * w)).squaredNorm();
}

// Do two triangles overlap by more than `eps`? Separating-axis test over the
// two normals, the nine edge-edge cross products and each triangle's in-plane
// edge normals (which separate coplanar ones). Any gap separates; a meeting
// of at most eps on an axis where both have width is touching, not overlap.
// Along a triangle's own normal it has no width, so such an axis cannot say
// "touching" (a piercing pair shows zero overlap there). Conservative for a
// needle (no width almost anywhere): it may count as overlapping a neighbour
// it only touches. Same test as analysis_tools/mesh_checks.h, kept apart so
// the checks there stay independent of this code.
bool triangles_overlap(const std::array<Eigen::Vector3d, 3>& a,
                       const std::array<Eigen::Vector3d, 3>& b, double eps)
{
    const Eigen::Vector3d na = (a[1] - a[0]).cross(a[2] - a[0]);
    const Eigen::Vector3d nb = (b[1] - b[0]).cross(b[2] - b[0]);
    std::array<Eigen::Vector3d, 17> axes;
    std::size_t count = 0;
    axes[count++] = na;
    axes[count++] = nb;
    for (int i = 0; i < 3; ++i)
    {
        const Eigen::Vector3d ea = a[(i + 1) % 3] - a[i];
        const Eigen::Vector3d eb = b[(i + 1) % 3] - b[i];
        axes[count++] = na.cross(ea);
        axes[count++] = nb.cross(eb);
        for (int j = 0; j < 3; ++j)
        {
            axes[count++] = ea.cross(b[(j + 1) % 3] - b[j]);
        }
    }
    bool touching = false;
    for (std::size_t k = 0; k < count; ++k)
    {
        // Whether the projections have a gap does not depend on the axis's
        // length, so with no touching allowance the axis is used as it is.
        Eigen::Vector3d u = axes[k];
        if (eps > 0.0)
        {
            const double length = u.norm();
            if (!(length > 1e-300))
            {
                continue;
            }
            u /= length;
        }
        else if (u.squaredNorm() == 0.0)
        {
            continue;
        }
        double a0 = infinity, a1 = -infinity, b0 = infinity, b1 = -infinity;
        for (int c = 0; c < 3; ++c)
        {
            a0 = std::min(a0, u.dot(a[c]));
            a1 = std::max(a1, u.dot(a[c]));
            b0 = std::min(b0, u.dot(b[c]));
            b1 = std::max(b1, u.dot(b[c]));
        }
        const double overlap = std::min(a1, b1) - std::max(a0, b0);
        if (overlap < 0.0)
        {
            return false;
        }
        if (overlap <= eps && a1 - a0 > eps && b1 - b0 > eps)
        {
            touching = true;
        }
    }
    return !touching;
}

} // namespace

SimplifyResult simplify_mesh(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<char>& locked_in,
    double tolerance,
    Eigen::MatrixXd& U,
    Eigen::MatrixXi& G,
    Eigen::VectorXi& I,
    const Eigen::RowVector3d* box_min,
    const Eigen::RowVector3d* box_max)
{
    SimplifyResult result;
    const bool boxed = box_min != nullptr && box_max != nullptr;
    const auto in_box = [&](const Eigen::Vector3d& q)
    {
        return !boxed || ((q.transpose().array() >= box_min->array()).all() &&
                          (q.transpose().array() <= box_max->array()).all());
    };
    result.faces_before = static_cast<std::size_t>(F.rows());
    result.faces_after = result.faces_before;
    const Eigen::Index n = V.rows();
    if (F.rows() < 4 || n < 4 || !(tolerance > 0.0) ||
        locked_in.size() != static_cast<std::size_t>(n))
    {
        result.outcome = SimplifyOutcome::TooSmall;
        leave_unchanged(V, F, U, G, I);
        return result;
    }

    // Close the boundary with a point at infinity (index n), as qslim() does.
    Eigen::MatrixXi FO;
    igl::connect_boundary_to_infinity(F, static_cast<int>(n), FO);
    Eigen::MatrixXd VO(n + 1, 3);
    VO.topRows(n) = V;
    VO.row(n).setConstant(infinity);
    if (!igl::is_edge_manifold(FO))
    {
        result.outcome = SimplifyOutcome::Refused;
        leave_unchanged(V, F, U, G, I);
        return result;
    }

    // Locked: as given, the point at infinity (so nothing on the boundary
    // moves), and any vertex whose faces do not form one fan (collapses
    // circulate around a vertex and would miss the other fans).
    std::vector<char> locked(static_cast<std::size_t>(n) + 1, 0);
    for (Eigen::Index v = 0; v < n; ++v)
    {
        locked[static_cast<std::size_t>(v)] = locked_in[static_cast<std::size_t>(v)] ? 1 : 0;
    }
    locked[static_cast<std::size_t>(n)] = 1;
    {
        Eigen::Matrix<bool, Eigen::Dynamic, 1> manifold;
        igl::is_vertex_manifold(FO, manifold);
        for (Eigen::Index v = 0; v < manifold.rows() && v < n; ++v)
        {
            if (!manifold(v))
            {
                locked[static_cast<std::size_t>(v)] = 1;
            }
        }
    }

    // One plane per original face, each counted once (not by area). A face
    // with no area has no plane and adds nothing.
    std::vector<Quadric> quadrics(static_cast<std::size_t>(n) + 1);
    for (Eigen::Index f = 0; f < F.rows(); ++f)
    {
        const Eigen::Vector3d a = V.row(F(f, 0)).transpose();
        const Eigen::Vector3d b = V.row(F(f, 1)).transpose();
        const Eigen::Vector3d c = V.row(F(f, 2)).transpose();
        const Eigen::Vector3d normal = area_normal(a, b, c);
        const double length = normal.norm();
        const double scale = (b - a).squaredNorm() + (c - a).squaredNorm() + (c - b).squaredNorm();
        if (!(length > 1e-12 * scale) || !std::isfinite(length))
        {
            continue;
        }
        const Eigen::Vector3d unit = normal / length;
        Quadric plane;
        plane.A = unit * unit.transpose();
        const double d = -unit.dot(a);
        plane.b = d * unit;
        plane.c = d * d;
        for (int corner = 0; corner < 3; ++corner)
        {
            Quadric& q = quadrics[static_cast<std::size_t>(F(f, corner))];
            q = q + plane;
        }
    }

    const double tolerance_squared = tolerance * tolerance;

    // The real error bound (the planes above are only a fast first filter and
    // the order of collapses; a vertex can be near every plane yet far from
    // the surface in the narrow wedge where two faces meet at a sharp angle):
    //   - every raw vertex and raw triangle centre stays within `tolerance` of
    //     the simplified surface: each is owned by one current triangle, and a
    //     collapse re-checks the points owned by the triangles it changes
    //     against the new ones (a point stays with its own triangle while that
    //     is close enough, else goes to the nearest close-enough one);
    //   - every new vertex lies within `tolerance` of the raw surface.
    igl::AABB<Eigen::MatrixXd, 3> raw_tree;
    raw_tree.init(V, F);
    std::vector<Eigen::Vector3d> samples;
    samples.reserve(static_cast<std::size_t>(n + F.rows()));
    for (Eigen::Index v = 0; v < n; ++v)
    {
        samples.push_back(V.row(v).transpose());
    }
    std::vector<std::vector<int>> owned(static_cast<std::size_t>(FO.rows()));
    std::vector<char> vertex_owned(static_cast<std::size_t>(n), 0);
    for (Eigen::Index f = 0; f < F.rows(); ++f)
    {
        const Eigen::Vector3d centre =
            (V.row(F(f, 0)) + V.row(F(f, 1)) + V.row(F(f, 2))).transpose() / 3.0;
        owned[static_cast<std::size_t>(f)].push_back(static_cast<int>(samples.size()));
        samples.push_back(centre);
        for (int corner = 0; corner < 3; ++corner)
        {
            const int v = F(f, corner);
            if (!vertex_owned[static_cast<std::size_t>(v)])
            {
                vertex_owned[static_cast<std::size_t>(v)] = 1;
                owned[static_cast<std::size_t>(f)].push_back(v);
            }
        }
    }
    // No collapse may make the surface cross itself. Where two surfaces come
    // within about twice the tolerance (the two faces of a thin sliver, a
    // wedge), the bounds above allow it, so each new triangle is tested
    // against the current triangles near it, found through a grid of buckets
    // that each triangle is listed in (kept up to date as triangles change).
    double mean_edge = 0.0;
    for (Eigen::Index f = 0; f < F.rows(); ++f)
    {
        mean_edge += (V.row(F(f, 0)) - V.row(F(f, 1))).norm();
    }
    mean_edge /= static_cast<double>(F.rows());
    // Bucket size: an eighth of the piece (a block of 8 coarse cells: one
    // coarse cell), kept between 2 and 8 mean edges. Tied to the raw edge
    // alone, buckets were so small that a large flat triangle late in the
    // run covered tens of thousands of them (half the time, measured). The
    // size only changes how candidates are found, never which are tested.
    const double extent = (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
    const double bucket = std::max(std::clamp(extent / 8.0, 2.0 * mean_edge, 8.0 * mean_edge), 1e-300);
    // No touching allowance: with nearly parallel edges some test axes come
    // from tiny cross products, and an allowance there let a real (6 um)
    // in-plane overlap pass as "touching". Strict, any overlap blocks; a
    // direction can never wrongly separate two triangles, so nothing real
    // gets through, at the cost of blocking a few collapses next to needles.
    const double touch = 0.0;
    const Eigen::RowVector3d origin = V.colwise().minCoeff();
    std::unordered_map<std::uint64_t, std::vector<int>> buckets;
    std::vector<std::vector<std::uint64_t>> face_buckets(static_cast<std::size_t>(FO.rows()));
    const auto bucket_key = [](std::int64_t x, std::int64_t y, std::int64_t z)
    {
        // 21 bits per axis (2 million buckets each way: far more than a block).
        const std::uint64_t mask = (std::uint64_t{1} << 21) - 1;
        return (static_cast<std::uint64_t>(x) & mask) | ((static_cast<std::uint64_t>(y) & mask) << 21) |
               ((static_cast<std::uint64_t>(z) & mask) << 42);
    };
    const auto for_buckets = [&](const std::array<Eigen::Vector3d, 3>& t, const auto& visit)
    {
        Eigen::Vector3d lo = t[0].cwiseMin(t[1]).cwiseMin(t[2]);
        Eigen::Vector3d hi = t[0].cwiseMax(t[1]).cwiseMax(t[2]);
        const Eigen::Vector3d o = origin.transpose();
        for (std::int64_t x = static_cast<std::int64_t>(std::floor((lo.x() - o.x()) / bucket));
             x <= static_cast<std::int64_t>(std::floor((hi.x() - o.x()) / bucket)); ++x)
            for (std::int64_t y = static_cast<std::int64_t>(std::floor((lo.y() - o.y()) / bucket));
                 y <= static_cast<std::int64_t>(std::floor((hi.y() - o.y()) / bucket)); ++y)
                for (std::int64_t z = static_cast<std::int64_t>(std::floor((lo.z() - o.z()) / bucket));
                     z <= static_cast<std::int64_t>(std::floor((hi.z() - o.z()) / bucket)); ++z)
                    visit(bucket_key(x, y, z));
    };
    const auto corners_of = [](const Eigen::MatrixXd& W, const Eigen::MatrixXi& FW, int f)
    {
        return std::array<Eigen::Vector3d, 3>{W.row(FW(f, 0)).transpose(), W.row(FW(f, 1)).transpose(),
                                               W.row(FW(f, 2)).transpose()};
    };
    const auto unlist = [&](int f)
    {
        for (const std::uint64_t key : face_buckets[static_cast<std::size_t>(f)])
        {
            std::vector<int>& list = buckets[key];
            list.erase(std::remove(list.begin(), list.end(), f), list.end());
        }
        face_buckets[static_cast<std::size_t>(f)].clear();
    };
    const auto list_face = [&](int f, const std::array<Eigen::Vector3d, 3>& t)
    {
        for_buckets(t, [&](std::uint64_t key)
        {
            buckets[key].push_back(f);
            face_buckets[static_cast<std::size_t>(f)].push_back(key);
        });
    };
    for (Eigen::Index f = 0; f < F.rows(); ++f)
    {
        list_face(static_cast<int>(f), corners_of(V, F, static_cast<int>(f)));
    }
    std::vector<std::uint32_t> stamp(static_cast<std::size_t>(FO.rows()), 0);
    std::uint32_t stamp_now = 0;
    std::vector<std::uint32_t> changed_mark(static_cast<std::size_t>(FO.rows()), 0);
    std::uint32_t collapse_now = 0;
    std::vector<int> candidates;

    // What the last accepted pre-collapse check worked out, applied in
    // post_collapse only if the collapse then really happens.
    bool pending = false;
    std::vector<int> pending_faces;
    std::vector<int> pending_kept;
    std::vector<std::pair<int, int>> pending_owner;    // (sample, face)

    // Cost and placement of collapsing edge e. Called from several threads at
    // first (libigl's own parallel loop), so it only reads.
    const igl::decimate_cost_and_placement_callback cost_and_placement =
        [&](const int e, const Eigen::MatrixXd& W, const Eigen::MatrixXi&,
            const Eigen::MatrixXi& E, const Eigen::VectorXi&, const Eigen::MatrixXi&,
            const Eigen::MatrixXi&, double& cost, Eigen::RowVectorXd& p)
    {
        p.resize(3);
        const int v1 = E(e, 0);
        const int v2 = E(e, 1);
        if (locked[static_cast<std::size_t>(v1)] || locked[static_cast<std::size_t>(v2)])
        {
            cost = infinity;
            p = W.row(v1);
            return;
        }
        const Quadric q = quadrics[static_cast<std::size_t>(v1)] +
            quadrics[static_cast<std::size_t>(v2)];
        const Eigen::Vector3d x1 = W.row(v1).transpose();
        const Eigen::Vector3d x2 = W.row(v2).transpose();
        const Eigen::Vector3d mid = 0.5 * (x1 + x2);
        const double half_length = 0.5 * (x2 - x1).norm();

        // The minimiser nearest the midpoint: mid - A+ (A mid + b), with the
        // pseudo-inverse dropping directions the planes leave free (all but
        // one on a flat area, one along a crease).
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(q.A);
        Eigen::Vector3d best = mid;
        if (solver.info() == Eigen::Success)
        {
            const Eigen::Vector3d lambda = solver.eigenvalues();
            const Eigen::Matrix3d Q = solver.eigenvectors();
            const double largest = std::max(lambda.cwiseAbs().maxCoeff(), 1e-300);
            const Eigen::Vector3d gradient = q.A * mid + q.b;
            Eigen::Vector3d step = Eigen::Vector3d::Zero();
            for (int k = 0; k < 3; ++k)
            {
                if (lambda(k) > 1e-6 * largest)
                {
                    step += Q.col(k) * (Q.col(k).dot(gradient) / lambda(k));
                }
            }
            best = mid - step;
        }
        // Too far from the edge (a nearly flat but not quite flat area makes
        // the system ill-conditioned), or outside the box: fall back to the
        // best of the two ends and the midpoint, the classic choice (inside
        // the box whenever the ends are, as it is convex).
        if (!best.allFinite() || (best - mid).norm() > half_length || !in_box(best))
        {
            best = mid;
            for (const Eigen::Vector3d& candidate : {x1, x2})
            {
                if (q(candidate) < q(best))
                {
                    best = candidate;
                }
            }
        }
        const double error = std::max(0.0, q(best));
        p = best.transpose();
        cost = (std::isfinite(error) && error <= tolerance_squared) ? error : infinity;
    };

    // Ends of the edge being collapsed, for merging their quadrics afterwards
    // (collapse_edge clears E(e, :)).
    int collapsing_v1 = -1;
    int collapsing_v2 = -1;

    const igl::decimate_pre_collapse_callback pre_collapse =
        [&](const Eigen::MatrixXd& W, const Eigen::MatrixXi& FW, const Eigen::MatrixXi& E,
            const Eigen::VectorXi& EMAP, const Eigen::MatrixXi& EF, const Eigen::MatrixXi& EI,
            const igl::min_heap<std::tuple<double, int, int>>&, const Eigen::VectorXi&,
            const Eigen::MatrixXd& C, const int e) -> bool
    {
        const int v1 = E(e, 0);
        const int v2 = E(e, 1);
        collapsing_v1 = v1;
        collapsing_v2 = v2;
        pending = false;
        if (locked[static_cast<std::size_t>(v1)] || locked[static_cast<std::size_t>(v2)])
        {
            return false;
        }
        const Eigen::Vector3d p = C.row(e).transpose();
        if (!p.allFinite() || !in_box(p))
        {
            return false;
        }
        // The new vertex must lie within the tolerance of the raw surface.
        {
            int face = -1;
            Eigen::RowVector3d closest;
            if (!(raw_tree.squared_distance(V, F, p.transpose(), face, closest) <= tolerance_squared))
            {
                return false;
            }
        }
        // No surrounding triangle may flip or become degenerate. The two faces
        // on the edge itself disappear and are skipped.
        std::vector<int> changed;                            // every face around either end
        std::vector<int> kept_faces;                         // ... except the two that disappear
        std::vector<std::array<Eigen::Vector3d, 3>> kept_after;
        // The fan's average normal (sum of area normals: the well-shaped
        // triangles dominate), the reference for faces that are needles now,
        // whose own normal says little but which must not be turned over.
        Eigen::Vector3d fan_normal = Eigen::Vector3d::Zero();
        for (const bool around_second : {false, true})
        {
            std::vector<int> ring_vertices;
            std::vector<int> ring_faces;
            igl::circulation(e, around_second, FW, EMAP, EF, EI, ring_vertices, ring_faces);
            for (const int f : ring_faces)
            {
                const int a = FW(f, 0), b = FW(f, 1), c = FW(f, 2);
                if (a == n || b == n || c == n)
                {
                    return false;
                }
                fan_normal += area_normal(W.row(a).transpose(), W.row(b).transpose(), W.row(c).transpose());
            }
        }
        for (const bool around_second : {false, true})
        {
            std::vector<int> ring_vertices;
            std::vector<int> ring_faces;
            igl::circulation(e, around_second, FW, EMAP, EF, EI, ring_vertices, ring_faces);
            for (const int f : ring_faces)
            {
                if (std::find(changed.begin(), changed.end(), f) != changed.end())
                {
                    continue;                                // around both ends: seen already
                }
                changed.push_back(f);
                const int a = FW(f, 0);
                const int b = FW(f, 1);
                const int c = FW(f, 2);
                const bool has1 = a == v1 || b == v1 || c == v1;
                const bool has2 = a == v2 || b == v2 || c == v2;
                if (has1 && has2)
                {
                    continue;
                }
                if (a == n || b == n || c == n)
                {
                    return false;    // touches the point at infinity: cannot happen unlocked, refuse
                }
                Eigen::Vector3d before[3] = {W.row(a).transpose(), W.row(b).transpose(),
                                             W.row(c).transpose()};
                Eigen::Vector3d after[3] = {before[0], before[1], before[2]};
                const int corners[3] = {a, b, c};
                for (int k = 0; k < 3; ++k)
                {
                    if (corners[k] == v1 || corners[k] == v2)
                    {
                        after[k] = p;
                    }
                }
                const Eigen::Vector3d nb = area_normal(before[0], before[1], before[2]);
                const Eigen::Vector3d na = area_normal(after[0], after[1], after[2]);
                if (!(quality(after[0], after[1], after[2]) >= minimum_quality))
                {
                    return false;                       // would become a needle (or degenerate)
                }
                // A well-shaped triangle may not turn by 60 degrees or more; a
                // needle's own normal says little, so it is held to the fan's
                // average instead: it may not end up facing against it.
                if (quality(before[0], before[1], before[2]) >= minimum_quality)
                {
                    if (na.dot(nb) < 0.5 * na.norm() * nb.norm())
                    {
                        return false;
                    }
                }
                else if (!(na.dot(fan_normal) > 0.0))
                {
                    return false;
                }
                kept_faces.push_back(f);
                kept_after.push_back({after[0], after[1], after[2]});
            }
        }
        if (kept_faces.empty())
        {
            return false;
        }
        // Every raw point owned by a changed face must stay within the
        // tolerance of the new faces. It stays with its own face if that face
        // remains and is still close enough (most do: the cheap case);
        // otherwise it goes to the nearest new face, if any is close enough.
        pending_faces = changed;
        pending_owner.clear();
        for (const int f : changed)
        {
            const auto own = std::find(kept_faces.begin(), kept_faces.end(), f);
            const std::size_t own_index = static_cast<std::size_t>(own - kept_faces.begin());
            for (const int sample : owned[static_cast<std::size_t>(f)])
            {
                const Eigen::Vector3d& q = samples[static_cast<std::size_t>(sample)];
                if (own != kept_faces.end() &&
                    point_triangle_squared_distance(q, kept_after[own_index][0], kept_after[own_index][1],
                                                    kept_after[own_index][2]) <= tolerance_squared)
                {
                    pending_owner.emplace_back(sample, f);
                    continue;
                }
                double best = infinity;
                int best_face = -1;
                for (std::size_t k = 0; k < kept_faces.size(); ++k)
                {
                    const double d = point_triangle_squared_distance(
                        q, kept_after[k][0], kept_after[k][1], kept_after[k][2]);
                    if (d < best)
                    {
                        best = d;
                        best_face = kept_faces[k];
                    }
                }
                if (!(best <= tolerance_squared))
                {
                    return false;
                }
                pending_owner.emplace_back(sample, best_face);
            }
        }
        // No new triangle may cross a triangle near it (other than the ones
        // changing, and the ones it shares a corner with). The candidates are
        // gathered once, from the buckets under the box around all the new
        // triangles, and each is tested against the new triangles whose own
        // boxes it meets.
        const int kept_vertex = std::min(v1, v2);
        ++collapse_now;
        for (const int f : changed)
        {
            changed_mark[static_cast<std::size_t>(f)] = collapse_now;
        }
        std::array<Eigen::Vector3d, 3> around = kept_after[0];
        std::vector<std::array<Eigen::Vector3d, 2>> boxes(kept_faces.size());
        for (std::size_t k = 0; k < kept_faces.size(); ++k)
        {
            boxes[k] = {kept_after[k][0].cwiseMin(kept_after[k][1]).cwiseMin(kept_after[k][2]),
                        kept_after[k][0].cwiseMax(kept_after[k][1]).cwiseMax(kept_after[k][2])};
            around[0] = around[0].cwiseMin(boxes[k][0]);
            around[1] = around[1].cwiseMax(boxes[k][1]);
        }
        around[2] = around[0];
        ++stamp_now;
        candidates.clear();
        for_buckets(around, [&](std::uint64_t key)
        {
            const auto found = buckets.find(key);
            if (found == buckets.end())
            {
                return;
            }
            for (const int g : found->second)
            {
                if (stamp[static_cast<std::size_t>(g)] != stamp_now)
                {
                    stamp[static_cast<std::size_t>(g)] = stamp_now;
                    if (changed_mark[static_cast<std::size_t>(g)] != collapse_now)
                    {
                        candidates.push_back(g);
                    }
                }
            }
        });
        for (const int g : candidates)
        {
            const int gi[3] = {FW(g, 0), FW(g, 1), FW(g, 2)};
            if (gi[0] == gi[1] && gi[1] == gi[2])
            {
                continue;                                    // removed by an earlier collapse
            }
            const std::array<Eigen::Vector3d, 3> other = corners_of(W, FW, g);
            const Eigen::Vector3d glo = other[0].cwiseMin(other[1]).cwiseMin(other[2]);
            const Eigen::Vector3d ghi = other[0].cwiseMax(other[1]).cwiseMax(other[2]);
            for (std::size_t k = 0; k < kept_faces.size(); ++k)
            {
                if ((glo.array() > boxes[k][1].array()).any() || (ghi.array() < boxes[k][0].array()).any())
                {
                    continue;                                // boxes apart: cannot cross
                }
                const int f = kept_faces[k];
                bool shares = false;
                for (int c = 0; c < 3; ++c)
                {
                    int id = FW(f, c);
                    if (id == v1 || id == v2)
                    {
                        id = kept_vertex;
                    }
                    shares = shares || id == gi[0] || id == gi[1] || id == gi[2];
                }
                if (!shares && triangles_overlap(kept_after[k], other, touch))
                {
                    return false;
                }
            }
        }
        pending_kept = kept_faces;
        pending = true;
        return true;
    };

    const igl::decimate_post_collapse_callback post_collapse =
        [&](const Eigen::MatrixXd& W, const Eigen::MatrixXi& FW, const Eigen::MatrixXi&,
            const Eigen::VectorXi&, const Eigen::MatrixXi&, const Eigen::MatrixXi&,
            const igl::min_heap<std::tuple<double, int, int>>&, const Eigen::VectorXi&,
            const Eigen::MatrixXd&, const int, const int, const int, const int, const int,
            const bool collapsed)
    {
        if (collapsed && collapsing_v1 >= 0 && collapsing_v2 >= 0)
        {
            // collapse_edge keeps the lower index.
            const int kept = std::min(collapsing_v1, collapsing_v2);
            const Quadric merged = quadrics[static_cast<std::size_t>(collapsing_v1)] +
                quadrics[static_cast<std::size_t>(collapsing_v2)];
            quadrics[static_cast<std::size_t>(kept)] = merged;
            if (pending)
            {
                for (const int f : pending_faces)
                {
                    owned[static_cast<std::size_t>(f)].clear();
                    unlist(f);                               // moved or removed
                }
                for (const auto& [sample, face] : pending_owner)
                {
                    owned[static_cast<std::size_t>(face)].push_back(sample);
                }
                for (const int f : pending_kept)
                {
                    list_face(f, corners_of(W, FW, f));     // at its new place
                }
            }
        }
        pending = false;
    };

    // Never stop early: the run ends when every remaining edge is locked or
    // over the tolerance (cost infinite).
    const igl::decimate_stopping_condition_callback never_stop =
        [](const Eigen::MatrixXd&, const Eigen::MatrixXi&, const Eigen::MatrixXi&,
           const Eigen::VectorXi&, const Eigen::MatrixXi&, const Eigen::MatrixXi&,
           const igl::min_heap<std::tuple<double, int, int>>&, const Eigen::VectorXi&,
           const Eigen::MatrixXd&, const int, const int, const int, const int, const int)
    {
        return false;
    };

    // decimate() returns false both for refused input (ruled out above) and
    // for its normal end (only infinite-cost edges left), so its return value
    // says nothing here; the output is checked below instead.
    Eigen::MatrixXd UW;
    Eigen::MatrixXi GW;
    Eigen::VectorXi JW;
    Eigen::VectorXi IW;
    igl::decimate(VO, FO, cost_and_placement, never_stop, pre_collapse, post_collapse,
                  UW, GW, JW, IW);

    // Drop the faces on the point at infinity (born from rows >= F.rows() of
    // FO) and the point itself, as qslim() does.
    std::vector<int> keep;
    keep.reserve(static_cast<std::size_t>(GW.rows()));
    for (Eigen::Index f = 0; f < GW.rows(); ++f)
    {
        if (JW(f) < F.rows())
        {
            keep.push_back(static_cast<int>(f));
        }
    }
    Eigen::MatrixXi GK(static_cast<Eigen::Index>(keep.size()), 3);
    for (std::size_t k = 0; k < keep.size(); ++k)
    {
        GK.row(static_cast<Eigen::Index>(k)) = GW.row(keep[k]);
    }
    Eigen::MatrixXd UK;
    Eigen::MatrixXi GF;
    Eigen::VectorXi unused;
    Eigen::VectorXi IK;
    igl::remove_unreferenced(UW, GK, UK, GF, unused, IK);
    Eigen::VectorXi IF(IK.rows());
    for (Eigen::Index u = 0; u < IK.rows(); ++u)
    {
        IF(u) = IW(IK(u));
    }

    // Check the promises before using the result; otherwise keep the input.
    bool sound = GF.rows() > 0 && UK.allFinite() && UK.rows() == IF.rows();
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    for (Eigen::Index u = 0; sound && u < IF.rows(); ++u)
    {
        const int origin = IF(u);
        if (origin < 0 || origin >= n || seen[static_cast<std::size_t>(origin)])
        {
            sound = false;
            break;
        }
        seen[static_cast<std::size_t>(origin)] = 1;
        if (locked_in[static_cast<std::size_t>(origin)] && !(UK.row(u) == V.row(origin)))
        {
            sound = false;
        }
    }
    for (Eigen::Index v = 0; sound && v < n; ++v)
    {
        if (locked_in[static_cast<std::size_t>(v)] && !seen[static_cast<std::size_t>(v)])
        {
            sound = false;    // a locked vertex disappeared
        }
    }
    if (!sound)
    {
        result.outcome = SimplifyOutcome::Refused;
        leave_unchanged(V, F, U, G, I);
        return result;
    }
    U = std::move(UK);
    G = std::move(GF);
    I = std::move(IF);
    result.outcome = SimplifyOutcome::Simplified;
    result.faces_after = static_cast<std::size_t>(G.rows());
    return result;
}
