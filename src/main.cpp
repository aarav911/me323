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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "fit_config.h"
#include "head_lid.h"
#include "helmet_fit.h"
#include "lattice_generator.h"

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
    LatticeType startup_lattice = LatticeType::Gyroid;
    bool generate_only = false;
    bool export_fit = false;
    std::string fit_override;          // "", "zoom" or "stretch"
    double lattice_room_target = 0.0;  // model units; 0 = no target in the report
    ShellMode shell_mode = ShellMode::Wall;
    double wall_skin_cells = 0.5;
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
        else if (option == "--lattice" && argument + 1 < argc)
        {
            const std::string requested = argv[++argument];
            if (requested == "auxetic")
            {
                startup_lattice = LatticeType::Auxetic;
            }
            else if (requested != "tpms")
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
                         " [--shell wall|band] [--shell-skin <cells>]\n";
            return 2;
        }
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
        if (!generate_final_helmet(
                fitted_head, F_head, head_lid.faces, fitted_helmet, F_helmet,
                type, generated, error, lattice_room_target, shell_mode,
                wall_skin_cells))
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