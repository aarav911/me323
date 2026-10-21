#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <imgui.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "coupon.h"
#include "fit_config.h"
#include "head_lid.h"
#include "helmet_fit.h"
#include "lattice_generator.h"
#include "print_export.h"

// A number with a fixed count of decimals, for messages.
std::string fixed_text(double value, int decimals)
{
    char text[64];
    std::snprintf(text, sizeof text, "%.*f", decimals, value);
    return text;
}

struct BoundingBox
{
    Eigen::Vector3d min;
    Eigen::Vector3d max;
    Eigen::Vector3d size;
    Eigen::Vector3d center;
};

constexpr const char* configuration_filename = "fit_config.txt";

BoundingBox get_bbox(const Eigen::MatrixXd& V)
{
    BoundingBox box;

    box.min = V.colwise().minCoeff();
    box.max = V.colwise().maxCoeff();

    box.size = box.max - box.min;
    box.center = (box.min + box.max) / 2.0;

    return box;
}

void fitted_vertices(
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    double gap,
    double head_scale,
    double helmet_scale,
    const Eigen::Vector3d& head_translation,
    const Eigen::Vector3d& helmet_translation_offset,
    Eigen::MatrixXd& head_vertices,
    Eigen::MatrixXd& helmet_vertices);

void fit_meshes(
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    const FitConfiguration& config,
    Eigen::MatrixXd& head_vertices,
    Eigen::MatrixXd& helmet_vertices);

void update_meshes(
    igl::opengl::glfw::Viewer& viewer,
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    const FitConfiguration& config)
{
    Eigen::MatrixXd head_vertices;
    Eigen::MatrixXd helmet_vertices;
    fit_meshes(
        original_head, original_helmet, original_head_box, original_helmet_box,
        automatic_helmet_scale, config, head_vertices, helmet_vertices);

    viewer.data_list[0].set_vertices(head_vertices);
    viewer.data_list[1].set_vertices(helmet_vertices);
    viewer.data_list[0].compute_normals();
    viewer.data_list[1].compute_normals();
}

void fitted_vertices(
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    double gap,
    double head_scale,
    double helmet_scale,
    const Eigen::Vector3d& head_translation,
    const Eigen::Vector3d& helmet_translation_offset,
    Eigen::MatrixXd& head_vertices,
    Eigen::MatrixXd& helmet_vertices)
{
    head_vertices = original_head;
    head_vertices.rowwise() -= original_head_box.center.transpose();
    head_vertices *= head_scale;
    head_vertices.rowwise() +=
        (original_head_box.center + head_translation).transpose();

    const double helmet_applied_scale = automatic_helmet_scale * helmet_scale;
    const double helmet_top =
        (original_helmet_box.max.y() - original_helmet_box.center.y())
        * helmet_applied_scale;
    const double head_top =
        original_head_box.center.y()
        + (original_head_box.max.y() - original_head_box.center.y()) * head_scale
        + head_translation.y();
    Eigen::RowVector3d helmet_translation;
    helmet_translation.x() =
        original_head_box.center.x() + head_translation.x()
        + helmet_translation_offset.x();
    helmet_translation.y() =
        head_top + gap - helmet_top + helmet_translation_offset.y();
    helmet_translation.z() =
        original_head_box.center.z() + head_translation.z()
        + helmet_translation_offset.z();

    helmet_vertices = original_helmet;
    helmet_vertices.rowwise() -= original_helmet_box.center.transpose();
    helmet_vertices *= helmet_applied_scale;
    helmet_vertices.rowwise() += helmet_translation;
}

// The head as always; the helmet by the configured fit. Zoom runs exactly the
// original fitted_vertices(); stretch replaces only the helmet (see
// fit_config.h), placed relative to the fitted head's box centre.
void fit_meshes(
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    const FitConfiguration& config,
    Eigen::MatrixXd& head_vertices,
    Eigen::MatrixXd& helmet_vertices)
{
    fitted_vertices(
        original_head, original_helmet, original_head_box, original_helmet_box,
        automatic_helmet_scale, config.gap, config.head_scale, config.helmet_scale,
        config.head_translation, config.helmet_translation,
        head_vertices, helmet_vertices);
    if (config.mode == FitMode::Stretch)
    {
        helmet_vertices = stretch_fit_helmet(
            original_helmet, original_helmet_box.center.transpose(),
            config.helmet_stretch, config.helmet_pitch_degrees,
            (original_head_box.center + config.head_translation).transpose(),
            config.helmet_offset);
    }
}


int main(int argc, char** argv)
{
    // Defaults (2026-10-21): the printable re-entrant helmet. --classic gives
    // back the original behaviour (Aarav's cell, coarse grid, no simplification).
    LatticeType startup_lattice = LatticeType::Auxetic;
    bool generate_only = false;
    bool export_fit = false;
    std::string fit_override;          // "", "zoom" or "stretch"
    double lattice_room_target = 0.0;  // model units; 0 = no target in the report
    ShellMode shell_mode = ShellMode::Wall;
    double wall_skin_cells = 0.5;
    double fine_spacing = 0.0004;      // model units; 0 = surface from the coarse grid
    bool fine_exact = false;           // tests: exact distances at every fine sample
    bool fine_all_blocks = false;      // tests: evaluate every block, skip none
    unsigned fine_block_cells = 8;     // tests: block size in coarse cells
    GenerationOptions cell_defaults;   // step 4b: the re-entrant cell and simplification
    AuxeticCell auxetic_cell = AuxeticCell::Reentrant;
    double cell_width = cell_defaults.cell_width;
    double cell_angle = cell_defaults.cell_angle_degrees;
    double cell_h_ratio = cell_defaults.cell_h_ratio;
    double wall_thickness = cell_defaults.wall_thickness;
    double line_width = cell_defaults.line_width;
    double simplify_tolerance = 0.00005;   // model units (0.05 mm); 0 = not printable mode
    double roof_chamfer = 0.0;             // degrees; 0 = none
    bool coupon = false;                   // write the two print-test coupons and stop
    // A number from argv[argument], or false (and a message) if it is not one
    // or fails the check.
    const auto read_number = [&](int at, double& value, bool (*ok)(double), const char* what)
    {
        char* end = nullptr;
        value = std::strtod(argv[at], &end);
        if (end == argv[at] || *end != '\0' || !std::isfinite(value) || !ok(value))
        {
            std::cerr << what << '\n';
            return false;
        }
        return true;
    };
    for (int argument = 1; argument < argc; ++argument)
    {
        const std::string option = argv[argument];
        if (option == "--generate-only")
        {
            generate_only = true;
        }
        else if (option == "--export-fit")
        {
            export_fit = true;
        }
        else if (option == "--fit" && argument + 1 < argc)
        {
            fit_override = argv[++argument];
            if (fit_override != "zoom" && fit_override != "stretch")
            {
                std::cerr << "Unknown fit: " << fit_override
                          << ". Choose 'zoom' or 'stretch'.\n";
                return 2;
            }
        }
        else if (option == "--lattice-room" && argument + 1 < argc)
        {
            char* end = nullptr;
            lattice_room_target = std::strtod(argv[++argument], &end);
            if (end == argv[argument] || *end != '\0' ||
                !std::isfinite(lattice_room_target) || lattice_room_target < 0.0)
            {
                std::cerr << "--lattice-room needs a non-negative number in the head "
                             "file's units (0.02 = 20 mm for a head in metres).\n";
                return 2;
            }
        }
        else if (option == "--shell" && argument + 1 < argc)
        {
            const std::string requested = argv[++argument];
            if (requested == "wall")
            {
                shell_mode = ShellMode::Wall;
            }
            else if (requested == "band")
            {
                shell_mode = ShellMode::Band;
            }
            else
            {
                std::cerr << "Unknown shell: " << requested
                          << ". Choose 'wall' or 'band'.\n";
                return 2;
            }
        }
        else if (option == "--shell-skin" && argument + 1 < argc)
        {
            char* end = nullptr;
            wall_skin_cells = std::strtod(argv[++argument], &end);
            if (end == argv[argument] || *end != '\0' ||
                !std::isfinite(wall_skin_cells) || wall_skin_cells < 0.0 ||
                wall_skin_cells > 3.0)
            {
                std::cerr << "--shell-skin needs a number of grid cells from 0 to 3.\n";
                return 2;
            }
        }
        else if (option == "--fine-spacing" && argument + 1 < argc)
        {
            char* end = nullptr;
            fine_spacing = std::strtod(argv[++argument], &end);
            if (end == argv[argument] || *end != '\0' ||
                !std::isfinite(fine_spacing) || fine_spacing < 0.0)
            {
                std::cerr << "--fine-spacing needs a non-negative number in the head "
                             "file's units (0.0005 = 0.5 mm for a head in metres; 0 = off).\n";
                return 2;
            }
        }
        else if (option == "--cell" && argument + 1 < argc)
        {
            const std::string requested = argv[++argument];
            if (requested == "original")
            {
                auxetic_cell = AuxeticCell::Original;
            }
            else if (requested == "reentrant")
            {
                auxetic_cell = AuxeticCell::Reentrant;
            }
            else
            {
                std::cerr << "Unknown cell: " << requested
                          << ". Choose 'original' or 'reentrant'.\n";
                return 2;
            }
        }
        else if (option == "--cell-width" && argument + 1 < argc)
        {
            if (!read_number(++argument, cell_width, [](double v) { return v > 0.0; },
                             "--cell-width needs a positive size in the head file's units "
                             "(0.010 = 10 mm for a head in metres)."))
            {
                return 2;
            }
        }
        else if (option == "--cell-angle" && argument + 1 < argc)
        {
            if (!read_number(++argument, cell_angle,
                             [](double v) { return v >= -60.0 && v <= -10.0; },
                             "--cell-angle needs degrees from -60 to -10 (negative = re-entrant)."))
            {
                return 2;
            }
        }
        else if (option == "--cell-h-ratio" && argument + 1 < argc)
        {
            if (!read_number(++argument, cell_h_ratio, [](double v) { return v >= 0.0; },
                             "--cell-h-ratio needs h / l > 0 (0 = the ideal-nu value)."))
            {
                return 2;
            }
        }
        else if (option == "--wall-thickness" && argument + 1 < argc)
        {
            if (!read_number(++argument, wall_thickness, [](double v) { return v > 0.0; },
                             "--wall-thickness needs a positive size in the head file's units "
                             "(0.0012 = 1.2 mm)."))
            {
                return 2;
            }
        }
        else if (option == "--line-width" && argument + 1 < argc)
        {
            if (!read_number(++argument, line_width, [](double v) { return v >= 0.0; },
                             "--line-width needs a size >= 0 in the head file's units "
                             "(0.0004 = 0.4 mm; 0 = no check)."))
            {
                return 2;
            }
        }
        else if (option == "--simplify" && argument + 1 < argc)
        {
            if (!read_number(++argument, simplify_tolerance, [](double v) { return v >= 0.0; },
                             "--simplify needs a tolerance >= 0 in the head file's units "
                             "(0.00005 = 0.05 mm; 0 = off)."))
            {
                return 2;
            }
        }
        else if (option == "--classic")
        {
            // The original behaviour; flags after this one still apply.
            auxetic_cell = AuxeticCell::Original;
            fine_spacing = 0.0;
            simplify_tolerance = 0.0;
        }
        else if (option == "--roof-chamfer" && argument + 1 < argc)
        {
            if (!read_number(++argument, roof_chamfer,
                             [](double v) { return v == 0.0 || (v >= 45.0 && v <= 80.0); },
                             "--roof-chamfer needs 0 (none) or 45-80 degrees from horizontal."))
            {
                return 2;
            }
        }
        else if (option == "--coupon")
        {
            coupon = true;
        }
        else if (option == "--fine-exact")
        {
            fine_exact = true;
        }
        else if (option == "--fine-all-blocks")
        {
            fine_all_blocks = true;
        }
        else if (option == "--fine-block-cells" && argument + 1 < argc)
        {
            char* end = nullptr;
            const long cells = std::strtol(argv[++argument], &end, 10);
            if (end == argv[argument] || *end != '\0' || cells < 1 || cells > 64)
            {
                std::cerr << "--fine-block-cells needs a whole number from 1 to 64.\n";
                return 2;
            }
            fine_block_cells = static_cast<unsigned>(cells);
        }
        else if (option == "--lattice" && argument + 1 < argc)
        {
            const std::string requested = argv[++argument];
            if (requested == "auxetic")
            {
                startup_lattice = LatticeType::Auxetic;
            }
            else if (requested == "tpms")
            {
                startup_lattice = LatticeType::Gyroid;    // also when it follows an earlier --lattice auxetic
            }
            else
            {
                std::cerr << "Unknown lattice case: " << requested
                          << ". Choose 'auxetic' or 'tpms'.\n";
                return 2;
            }
        }
        else
        {
            std::cerr << "Usage: me323 [--lattice auxetic|tpms] [--generate-only]"
                         " [--export-fit] [--fit zoom|stretch] [--lattice-room <size>]"
                         " [--shell wall|band] [--shell-skin <cells>]"
                         " [--fine-spacing <size>]\n"
                         "       [--cell original|reentrant] [--cell-width <size>]"
                         " [--cell-angle <deg>] [--cell-h-ratio <h/l>]\n"
                         "       [--wall-thickness <size>] [--line-width <size>]"
                         " [--simplify <tolerance>]\n"
                         "       [--roof-chamfer <deg>] [--classic] [--coupon]\n"
                         "       test options: [--fine-exact] [--fine-all-blocks]"
                         " [--fine-block-cells <n>]\n";
            return 2;
        }
    }

    // --coupon: the two print-test coupons (no head or helmet needed). The
    // cell flags are in the head file's units, metres for our scan; the
    // coupons are made in millimetres.
    if (coupon)
    {
        if (auxetic_cell != AuxeticCell::Reentrant)
        {
            std::cerr << "--coupon tests the re-entrant cell; drop --cell original / --classic.\n";
            return 2;
        }
        std::string error;
        ReentrantCell cell_mm;
        if (!make_reentrant_cell(1000.0 * cell_width, cell_angle, cell_h_ratio, 1000.0 * wall_thickness,
                                 cell_mm, error))
        {
            std::cerr << "Coupon: " << error << '\n';
            return 1;
        }
        if (cell_mm.width < 2.0 || cell_mm.width > 50.0)
        {
            std::cerr << "Coupon: a cell " << cell_mm.width << " mm wide; the cell sizes are taken "
                         "to be in metres (e.g. --cell-width 0.010).\n";
            return 1;
        }
        CouponSpec spec;
        spec.spacing = 1000.0 * (fine_spacing > 0.0 ? fine_spacing : 0.0004);
        spec.simplify_tolerance = 1000.0 * (simplify_tolerance > 0.0 ? simplify_tolerance : 0.00005);
        const double chamfer_b = roof_chamfer > 0.0 ? roof_chamfer : 55.0;
        for (const double chamfer : {0.0, chamfer_b})
        {
            spec.roof_chamfer_degrees = chamfer;
            const std::string name = chamfer > 0.0 ? "coupon_B_chamfer.stl" : "coupon_A_no_chamfer.stl";
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            CouponReport report;
            if (!generate_coupon(cell_mm, spec, V, F, report, error))
            {
                std::cerr << name << ": " << error << '\n';
                return 1;
            }
            if (!igl::writeSTL(name, V, F, igl::FileEncoding::Binary))
            {
                std::cerr << "Could not write " << name << ".\n";
                return 1;
            }
            std::cout << name << ": " << spec.width << " x " << spec.depth << " x " << report.height
                      << " mm (floor " << spec.floor << ", walls " << spec.walls_height << ", roof "
                      << spec.roof << ", sides " << spec.side << "), walls " << cell_mm.wall_thickness
                      << " mm, grid " << spec.spacing << " mm; "
                      << (chamfer > 0.0 ? "roof on " + fixed_text(chamfer, 0) + " deg slopes reaching " +
                                              fixed_text(report.chamfer_reach, 2) + " mm from the walls"
                                        : std::string("roof bridging the cells"))
                      << "; " << F.rows() << " triangles, " << report.fragments.pockets
                      << " sealed cells, " << report.fragments.removed << " fragments removed"
                      << (report.fragments.large_extra > 0 ? ", WARNING: separate pieces" : "") << ".\n";
        }
        std::cout << "Print both in TPU (walls vertical, as written). If coupon A's roof is flat and "
                     "solid, print the helmet as it is; if it sags, regenerate the helmet with "
                     "--roof-chamfer " << chamfer_b << ".\n";
        return 0;
    }

    // Desired clearance uses the same units as the head STL (e.g. 0.005
    // for 5 mm if the model is in meters, or 5.0 if it is in millimeters).
    FitConfiguration config;

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

    // Which head triangles only close the scan. Read from the file afresh: the
    // STL reader has consumed head_file.
    HeadLid head_lid;
    {
        std::string error;
        if (!read_head_lid("data/head.stl", V_head, F_head, head_lid, error))
        {
            std::cerr << error << '\n';
            return 1;
        }
        std::cout << head_lid.note << '\n';
    }

    const BoundingBox head = get_bbox(V_head);
    const BoundingBox helmet = get_bbox(V_helmet);
    if (head.size.y() <= 0.0 || helmet.size.y() <= 0.0)
    {
        std::cerr << "Cannot fit meshes with zero height.\n";
        return 1;
    }
    std::string config_status = "Configuration file: fit_config.txt";
    {
        std::string error;
        if (load_configuration(config, configuration_filename, error))
        {
            std::cout << "Loaded fit configuration from "
                      << configuration_filename << '\n';
            config_status = "Loaded fit_config.txt";
        }
        else if (error != "Could not open fit_config.txt.")
        {
            std::cerr << error << '\n';
            config_status = error;
        }
    }
    if (fit_override == "stretch")
    {
        if (!config.has_stretch())
        {
            std::cerr << "--fit stretch needs helmet_stretch in fit_config.txt (version 2); "
                         "write one with analysis_tools/stretch_fit_study.py --write-config.\n";
            return 2;
        }
        config.mode = FitMode::Stretch;
    }
    else if (fit_override == "zoom")
    {
        config.mode = FitMode::Zoom;
    }
    double automatic_helmet_scale =
        (head.size.y() + 2.0 * config.gap) / helmet.size.y();

    if (config.mode == FitMode::Stretch)
    {
        const Eigen::Vector3d fitted_size = helmet.size.cwiseProduct(config.helmet_stretch);
        std::cout << "\nSTRETCH FIT (fit_config.txt)\n"
                  << "  stretch X Y Z: " << config.helmet_stretch.transpose()
                  << "  (ratio " << config.helmet_stretch.maxCoeff() / config.helmet_stretch.minCoeff()
                  << ")\n  tilt: " << config.helmet_pitch_degrees << " deg (+ = front up)\n"
                  << "  helmet centre - head centre: " << config.helmet_offset.transpose()
                  << "\n  helmet box before tilting: " << fitted_size.transpose() << '\n';
    }

    std::cout << "\n====================================\n";
    std::cout << "AUTOMATIC HELMET ALIGNMENT\n";
    std::cout << "====================================\n";

    std::cout << "Head size:\n"
              << head.size.transpose()
              << "\n";

    std::cout << "\nOriginal helmet size:\n"
              << helmet.size.transpose()
              << "\n";

    const char* zoom_only = config.mode == FitMode::Stretch
        ? " (zoom fit only; not used by the stretch fit)" : "";
    std::cout << "\nInitial automatic helmet scale" << zoom_only << ":\n"
              << automatic_helmet_scale
              << "\n";

    std::cout << "\nGap" << zoom_only << ":\n"
              << config.gap
              << "\n";

    // ============================================================
    // VIEWER
    // ============================================================

    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(
        V_head,
        F_head
    );
    viewer.data().show_lines = false;

    viewer.append_mesh();
    viewer.data_list[1].set_mesh(
        V_helmet,
        F_helmet
    );
    viewer.data_list[1].show_lines = false;

    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);

    std::size_t final_mesh_index = 0;
    auto show_fit_meshes = [&]()
    {
        if (final_mesh_index != 0)
        {
            viewer.data_list[final_mesh_index].set_visible(false);
        }
        viewer.data_list[0].set_visible(true);
        viewer.data_list[1].set_visible(true);
    };

    auto generate_case = [&](LatticeType type) -> bool
    {
        Eigen::MatrixXd fitted_head;
        Eigen::MatrixXd fitted_helmet;
        fit_meshes(
            V_head, V_helmet, head, helmet, automatic_helmet_scale, config,
            fitted_head, fitted_helmet);

        // The exact head and helmet the generator receives, so external checks
        // (tools/check_output.py) use this geometry rather than re-deriving the
        // fit. Written before generation, so they exist even if it fails. The
        // head keeps its lid marker: triangle order is unchanged, so the
        // marked range still holds.
        if (export_fit)
        {
            std::string header_error;
            if (!igl::writeSTL("fitted_head.stl", fitted_head, F_head,
                               igl::FileEncoding::Binary) ||
                !igl::writeSTL("fitted_helmet.stl", fitted_helmet, F_helmet,
                               igl::FileEncoding::Binary))
            {
                config_status = "Could not write fitted_head.stl / fitted_helmet.stl.";
                std::cerr << config_status << '\n';
                return false;
            }
            if (!head_lid.faces.empty() &&
                !write_stl_header("fitted_head.stl", head_lid.header, header_error))
            {
                config_status = header_error;
                std::cerr << config_status << '\n';
                return false;
            }
        }

        GeneratedMesh generated;
        std::string error;
        GenerationOptions options;
        options.lattice_room_target = lattice_room_target;
        options.shell_mode = shell_mode;
        options.wall_skin_cells = wall_skin_cells;
        options.fine_spacing = fine_spacing;
        options.fine_exact = fine_exact;
        options.fine_all_blocks = fine_all_blocks;
        options.fine_block_cells = fine_block_cells;
        options.auxetic_cell = auxetic_cell;
        options.cell_width = cell_width;
        options.cell_angle_degrees = cell_angle;
        options.cell_h_ratio = cell_h_ratio;
        options.wall_thickness = wall_thickness;
        options.line_width = line_width;
        options.simplify_tolerance = simplify_tolerance;
        options.roof_chamfer_degrees = roof_chamfer;
        if (!generate_final_helmet(
                fitted_head, F_head, head_lid.faces, fitted_helmet, F_helmet,
                type, generated, error, options))
        {
            config_status = error;
            std::cerr << "Lattice generation failed: " << error << '\n';
            return false;
        }

        const bool auxetic = type == LatticeType::Auxetic;
        const std::string filename = auxetic
            ? "final_helmet_auxetic.stl"
            : "final_helmet_tpms.stl";
        if (!igl::writeSTL(
                filename, generated.vertices, generated.faces,
                igl::FileEncoding::Binary))
        {
            config_status = "Could not write " + filename + ".";
            std::cerr << config_status << '\n';
            return false;
        }
        // Printable mode: also the copy a slicer wants (mm, Z up, crown-down,
        // on the bed). The viewer keeps showing the model as fitted.
        if (simplify_tolerance > 0.0)
        {
            const double head_height =
                fitted_head.col(1).maxCoeff() - fitted_head.col(1).minCoeff();
            Eigen::MatrixXd print_vertices;
            PrintPlacement placement;
            make_print_copy(generated.vertices, head_height, print_vertices, placement);
            const std::string print_name = filename.substr(0, filename.size() - 4) + "_print.stl";
            if (!igl::writeSTL(print_name, print_vertices, generated.faces, igl::FileEncoding::Binary))
            {
                config_status = "Could not write " + print_name + ".";
                std::cerr << config_status << '\n';
                return false;
            }
            std::cout << "Wrote " << print_name << " for printing: x" << placement.scale
                      << " (to mm), crown-down (turned about X), on the bed: "
                      << placement.size.x() << " x " << placement.size.y() << " x "
                      << placement.size.z() << " mm.\n";
        }

        if (final_mesh_index != 0)
        {
            viewer.data_list[final_mesh_index].set_visible(false);
        }
        viewer.data_list[0].set_visible(false);
        viewer.data_list[1].set_visible(false);
        viewer.append_mesh();
        final_mesh_index = viewer.data_list.size() - 1;
        viewer.data_list[final_mesh_index].set_mesh(
            generated.vertices, generated.faces);
        viewer.data_list[final_mesh_index].show_lines = false;
        viewer.core().align_camera_center(generated.vertices, generated.faces);

        config_status = "Generated " + filename + " ("
            + std::to_string(generated.vertices.rows()) + " vertices, voxel "
            + std::to_string(generated.voxel_size) + ").";
        std::cout << config_status << '\n';
        return true;
    };

    menu.callback_draw_custom_window = [&]()
    {
        ImGui::SetNextWindowSize(ImVec2(420.0f, 520.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Head / helmet fit", nullptr, ImGuiWindowFlags_NoSavedSettings);
        ImGui::TextUnformatted("Translations use the head STL's units.");
        bool apply_configuration = false;

        // Fit mode. Switching to stretch with no stretch values yet starts
        // from the current zoom fit, so the helmet does not jump.
        int mode = config.mode == FitMode::Stretch ? 1 : 0;
        const int previous_mode = mode;
        ImGui::RadioButton("Zoom fit", &mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Stretch fit", &mode, 1);
        if (mode != previous_mode)
        {
            if (mode == 1 && !config.has_stretch())
            {
                Eigen::MatrixXd zoom_head;
                Eigen::MatrixXd zoom_helmet;
                FitConfiguration zoom = config;
                zoom.mode = FitMode::Zoom;
                fit_meshes(V_head, V_helmet, head, helmet, automatic_helmet_scale,
                           zoom, zoom_head, zoom_helmet);
                config.helmet_stretch = Eigen::Vector3d::Constant(
                    automatic_helmet_scale * config.helmet_scale);
                config.helmet_pitch_degrees = 0.0;
                config.helmet_offset = get_bbox(zoom_helmet).center
                    - (head.center + config.head_translation);
            }
            config.mode = mode == 1 ? FitMode::Stretch : FitMode::Zoom;
            apply_configuration = true;
        }

        ImGui::SliderFloat("Head scale", &config.head_scale, 0.1f, 3.0f, "%.3fx");
        apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();

        const double translation_step = head.size.y() / 1000.0;
        const double translation_step_fast = head.size.y() / 100.0;
        ImGui::Separator();
        ImGui::TextUnformatted("Head translation (X, Y, Z)");
        ImGui::InputScalarN(
            "Head position", ImGuiDataType_Double, config.head_translation.data(), 3,
            &translation_step, &translation_step_fast, "%.6g");
        apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::Separator();
        if (config.mode == FitMode::Zoom)
        {
            ImGui::Text("Helmet auto-scale: %.6g", automatic_helmet_scale);
            ImGui::SliderFloat(
                "Helmet scale", &config.helmet_scale, 0.1f, 3.0f, "%.3fx");
            apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::TextUnformatted("Helmet translation (X, Y, Z)");
            ImGui::InputScalarN(
                "Helmet position", ImGuiDataType_Double,
                config.helmet_translation.data(), 3,
                &translation_step, &translation_step_fast, "%.6g");
            apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();

            ImGui::InputDouble("Gap", &config.gap, translation_step,
                               translation_step_fast, "%.6g");
            apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();
        }
        else
        {
            ImGui::TextUnformatted("Helmet stretch per axis (X, Y, Z)");
            const double stretch_step = 0.001 * config.helmet_stretch.maxCoeff();
            const double stretch_step_fast = 0.01 * config.helmet_stretch.maxCoeff();
            ImGui::InputScalarN(
                "Stretch", ImGuiDataType_Double, config.helmet_stretch.data(), 3,
                &stretch_step, &stretch_step_fast, "%.6g");
            apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();
            const double pitch_step = 0.5;
            const double pitch_step_fast = 5.0;
            ImGui::InputScalar(
                "Tilt (deg, + = front up)", ImGuiDataType_Double,
                &config.helmet_pitch_degrees, &pitch_step, &pitch_step_fast, "%.3g");
            apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();
            ImGui::TextUnformatted("Helmet centre minus head centre (X, Y, Z)");
            ImGui::InputScalarN(
                "Helmet offset", ImGuiDataType_Double, config.helmet_offset.data(), 3,
                &translation_step, &translation_step_fast, "%.6g");
            apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();
            // Keep the values usable: positive stretch, tilt within +-90.
            for (int axis = 0; axis < 3; ++axis)
            {
                if (!(config.helmet_stretch[axis] > 0.0))
                {
                    config.helmet_stretch[axis] = automatic_helmet_scale;
                }
            }
            config.helmet_pitch_degrees =
                std::clamp(config.helmet_pitch_degrees, -90.0, 90.0);
        }

        if (ImGui::Button("Reset fit"))
        {
            config = FitConfiguration{};
            apply_configuration = true;
        }

        if (ImGui::Button("Save configuration"))
        {
            std::string error;
            if (save_configuration(config, configuration_filename, error))
            {
                std::cout << "Saved fit configuration to "
                          << configuration_filename << '\n';
                config_status = "Saved to fit_config.txt";
            }
            else
            {
                std::cerr << error << '\n';
                config_status = error;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load configuration"))
        {
            std::string error;
            if (load_configuration(config, configuration_filename, error))
            {
                apply_configuration = true;
                std::cout << "Loaded fit configuration from "
                          << configuration_filename << '\n';
                config_status = "Loaded fit_config.txt";
            }
            else
            {
                std::cerr << error << '\n';
                config_status = error;
            }
        }
        ImGui::TextWrapped("%s", config_status.c_str());

        if (apply_configuration)
        {
            automatic_helmet_scale =
                (head.size.y() + 2.0 * config.gap) / helmet.size.y();
            show_fit_meshes();
            update_meshes(
                viewer, V_head, V_helmet, head, helmet,
                automatic_helmet_scale, config);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Generate and export final helmet");
        if (ImGui::Button("Generate auxetic STL"))
        {
            generate_case(LatticeType::Auxetic);
        }
        ImGui::SameLine();
        if (ImGui::Button("Generate TPMS STL"))
        {
            generate_case(LatticeType::Gyroid);
        }
        if (ImGui::Button("Preview head and helmet fit"))
        {
            show_fit_meshes();
            config_status = "Showing the fitted input meshes.";
        }
        ImGui::End();
    };

    update_meshes(
        viewer, V_head, V_helmet, head, helmet, automatic_helmet_scale, config);

    if (!generate_case(startup_lattice))
    {
        return 1;
    }
    if (generate_only)
    {
        return 0;
    }
    viewer.launch();

    return 0;
}