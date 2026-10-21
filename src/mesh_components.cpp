#include "mesh_components.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace
{

int find_root(std::vector<int>& parent, int x)
{
    while (parent[static_cast<std::size_t>(x)] != x)
    {
        parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
        x = parent[static_cast<std::size_t>(x)];
    }
    return x;
}

} // namespace

void remove_fragments(
    Eigen::MatrixXd& vertices,
    Eigen::MatrixXi& faces,
    double relative_threshold,
    FragmentReport& report)
{
    report = FragmentReport{};
    const Eigen::Index m = faces.rows();
    if (m == 0)
    {
        return;
    }

    // Faces sharing an edge belong together: sort the (edge, face) pairs and
    // join consecutive faces with the same edge.
    std::vector<std::pair<std::uint64_t, int>> edges;
    edges.reserve(static_cast<std::size_t>(m) * 3);
    for (Eigen::Index f = 0; f < m; ++f)
    {
        for (int k = 0; k < 3; ++k)
        {
            const std::uint32_t a = static_cast<std::uint32_t>(faces(f, k));
            const std::uint32_t b = static_cast<std::uint32_t>(faces(f, (k + 1) % 3));
            edges.emplace_back((static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b),
                               static_cast<int>(f));
        }
    }
    std::sort(edges.begin(), edges.end());
    std::vector<int> parent(static_cast<std::size_t>(m));
    std::iota(parent.begin(), parent.end(), 0);
    for (std::size_t i = 1; i < edges.size(); ++i)
    {
        if (edges[i].first == edges[i - 1].first)
        {
            const int a = find_root(parent, edges[i].second);
            const int b = find_root(parent, edges[i - 1].second);
            if (a != b)
            {
                parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);    // lowest face is the root
            }
        }
    }

    // Signed volume per piece (tetrahedra to the mesh's centre, to keep the
    // sums well conditioned far from the origin).
    const Eigen::Vector3d centre = (0.5 * (vertices.colwise().minCoeff() + vertices.colwise().maxCoeff())).transpose();
    std::vector<int> piece_of(static_cast<std::size_t>(m));
    std::vector<int> root_index(static_cast<std::size_t>(m), -1);
    std::vector<double> volume;
    for (Eigen::Index f = 0; f < m; ++f)
    {
        const int root = find_root(parent, static_cast<int>(f));
        if (root_index[static_cast<std::size_t>(root)] < 0)
        {
            root_index[static_cast<std::size_t>(root)] = static_cast<int>(volume.size());
            volume.push_back(0.0);
        }
        const int piece = root_index[static_cast<std::size_t>(root)];
        piece_of[static_cast<std::size_t>(f)] = piece;
        const Eigen::Vector3d a = vertices.row(faces(f, 0)).transpose() - centre;
        const Eigen::Vector3d b = vertices.row(faces(f, 1)).transpose() - centre;
        const Eigen::Vector3d c = vertices.row(faces(f, 2)).transpose() - centre;
        volume[static_cast<std::size_t>(piece)] += a.dot(b.cross(c)) / 6.0;
    }
    report.pieces = volume.size();

    // The main body: the solid piece of largest volume (the first one on a tie).
    std::size_t main = 0;
    for (std::size_t p = 1; p < volume.size(); ++p)
    {
        if (volume[p] > volume[main])
        {
            main = p;
        }
    }
    report.main_volume = volume[main];
    const double threshold = relative_threshold * std::max(volume[main], 0.0);
    std::vector<char> keep(volume.size(), 1);
    for (std::size_t p = 0; p < volume.size(); ++p)
    {
        if (p == main)
        {
            continue;
        }
        if (volume[p] < 0.0)
        {
            ++report.pockets;
            report.pocket_volume -= volume[p];
        }
        else if (volume[p] < threshold)
        {
            keep[p] = 0;
            ++report.removed;
            report.removed_volume += volume[p];
        }
        else
        {
            ++report.large_extra;
            report.large_extra_volume += volume[p];
        }
    }
    if (report.removed == 0)
    {
        return;
    }

    // Keep the faces of kept pieces, in order, and only the vertices they use.
    std::vector<int> new_index(static_cast<std::size_t>(vertices.rows()), -1);
    Eigen::MatrixXi kept_faces(m, 3);
    Eigen::Index count = 0;
    int next = 0;
    for (Eigen::Index f = 0; f < m; ++f)
    {
        if (!keep[static_cast<std::size_t>(piece_of[static_cast<std::size_t>(f)])])
        {
            continue;
        }
        for (int k = 0; k < 3; ++k)
        {
            int& index = new_index[static_cast<std::size_t>(faces(f, k))];
            if (index < 0)
            {
                index = next++;
            }
            kept_faces(count, k) = index;
        }
        ++count;
    }
    Eigen::MatrixXd kept_vertices(next, 3);
    for (Eigen::Index v = 0; v < vertices.rows(); ++v)
    {
        if (new_index[static_cast<std::size_t>(v)] >= 0)
        {
            kept_vertices.row(new_index[static_cast<std::size_t>(v)]) = vertices.row(v);
        }
    }
    vertices = std::move(kept_vertices);
    faces = kept_faces.topRows(count).eval();
}
