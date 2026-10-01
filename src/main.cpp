#include <igl/opengl/glfw/Viewer.h>
#include <igl/readSTL.h>

#include <Eigen/Core>

#include <fstream>
#include <iostream>


struct BoundingBox
{
    Eigen::Vector3d min;
    Eigen::Vector3d max;
    Eigen::Vector3d size;
    Eigen::Vector3d center;
};


BoundingBox get_bbox(const Eigen::MatrixXd& V)
{
    BoundingBox box;

    box.min = V.colwise().minCoeff();
    box.max = V.colwise().maxCoeff();

    box.size = box.max - box.min;
    box.center = (box.min + box.max) / 2.0;

    return box;
}


int main()
{
    // ============================================================
    // USER PARAMETERS
    // ============================================================

    // Desired physical clearance between head and helmet.
    //
    // If your head STL is in meters:
    //     0.005 = 5 mm
    //
    // If your head STL is in mm:
    //     5.0 = 5 mm
    //
    double gap = 0.005;


    // Additional manual positioning.
    //
    // These are deliberately separate from the automatic alignment.
    //
    double offset_x = 0.0;
    double offset_y = 0.0;
    double offset_z = 0.0;


    // ============================================================
    // LOAD MESHES
    // ============================================================

    Eigen::MatrixXd V_head, V_helmet;
    Eigen::MatrixXi F_head, F_helmet;
    Eigen::MatrixXd N_head, N_helmet;

    std::ifstream head_file(
        "data/head.stl",
        std::ios::binary
    );

    std::ifstream helmet_file(
        "data/helmet.STL",
        std::ios::binary
    );

    if (!head_file)
    {
        std::cerr << "Failed to open head.stl\n";
        return 1;
    }

    if (!helmet_file)
    {
        std::cerr << "Failed to open helmet.stl\n";
        return 1;
    }

    if (!igl::readSTL(
        head_file,
        V_head,
        F_head,
        N_head))
    {
        std::cerr << "Failed to read head.stl\n";
        return 1;
    }

    if (!igl::readSTL(
        helmet_file,
        V_helmet,
        F_helmet,
        N_helmet))
    {
        std::cerr << "Failed to read helmet.stl\n";
        return 1;
    }


    // ============================================================
    // COMPUTE BOUNDING BOXES
    // ============================================================

    BoundingBox head = get_bbox(V_head);
    BoundingBox helmet = get_bbox(V_helmet);


    // ============================================================
    // AUTOMATIC SCALE
    //
    // Match helmet HEIGHT to head HEIGHT.
    //
    // We add 2*gap because the helmet should extend slightly
    // beyond the head.
    // ============================================================

    double desired_helmet_height =
        head.size.y() + 2.0 * gap;

    double scale =
        desired_helmet_height / helmet.size.y();


    // Apply uniform scale
    V_helmet *= scale;


    // Recalculate helmet bounding box
    helmet = get_bbox(V_helmet);


    // ============================================================
    // AUTOMATIC ALIGNMENT
    //
    // X:
    //     Center helmet on head.
    //
    // Z:
    //     Center helmet on head.
    //
    // Y:
    //     Align the TOP of helmet with the TOP of head,
    //     while maintaining the gap.
    // ============================================================

    Eigen::Vector3d translation;

    translation.x() =
        head.center.x() - helmet.center.x();

    translation.z() =
        head.center.z() - helmet.center.z();

    translation.y() =
        head.max.y() + gap - helmet.max.y();


    // Add user-controlled offsets
    translation.x() += offset_x;
    translation.y() += offset_y;
    translation.z() += offset_z;


    // Apply translation
    V_helmet.rowwise() += translation.transpose();


    // ============================================================
    // PRINT TRANSFORMATION
    // ============================================================

    std::cout << "\n====================================\n";
    std::cout << "AUTOMATIC HELMET ALIGNMENT\n";
    std::cout << "====================================\n";

    std::cout << "Head size:\n"
              << head.size.transpose()
              << "\n";

    std::cout << "\nOriginal helmet size:\n"
              << (helmet.size / scale).transpose()
              << "\n";

    std::cout << "\nHelmet size after scaling:\n"
              << helmet.size.transpose()
              << "\n";

    std::cout << "\nAutomatic scale:\n"
              << scale
              << "\n";

    std::cout << "\nGap:\n"
              << gap
              << "\n";

    std::cout << "\nTranslation:\n"
              << translation.transpose()
              << "\n";


    // ============================================================
    // VIEWER
    // ============================================================

    igl::opengl::glfw::Viewer viewer;

    // Head
    viewer.data().set_mesh(
        V_head,
        F_head
    );

    viewer.data().show_lines = false;


    // Helmet
    viewer.append_mesh();

    viewer.data_list[1].set_mesh(
        V_helmet,
        F_helmet
    );

    viewer.data_list[1].show_lines = false;


    // Launch
    viewer.launch();

    return 0;
}