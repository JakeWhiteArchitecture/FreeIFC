/* webifc_bind.cpp — pybind11 bindings for web-ifc geometry engine.
 *
 * Exposes a minimal Python API for FreeIFC to load IFC files and stream
 * per-element triangle meshes without subprocess calls or ifcopenshell.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "../web-ifc-src/src/cpp/web-ifc/modelmanager/ModelManager.h"

namespace py = pybind11;

// ── Module-level state ─────────────────────────────────────────────────────

static webifc::manager::ModelManager g_manager(false);  // MT_ENABLED=false
static std::mutex g_mutex;

// ── Helpers ────────────────────────────────────────────────────────────────

static constexpr int VFORMAT = 6;  // floats per vertex: xyz + normal

// ── open_model ─────────────────────────────────────────────────────────────

static int open_model(const std::string &path)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    webifc::manager::LoaderSettings settings;
    uint32_t modelID = g_manager.CreateModel(settings);

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        g_manager.CloseModel(modelID);
        throw std::runtime_error("Cannot open file: " + path);
    }

    g_manager.GetIfcLoader(modelID)->LoadFile(ifs);
    return static_cast<int>(modelID);
}

// ── stream_all_meshes ──────────────────────────────────────────────────────

static void stream_all_meshes(int modelID, py::function callback)
{
    uint32_t mid = static_cast<uint32_t>(modelID);

    // Collect express IDs under the GIL, then release for C++ work.
    std::vector<uint32_t> expressIds;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_manager.IsModelOpen(mid))
            throw std::runtime_error("Model not open");
        expressIds = g_manager.GetIfcLoader(mid)->GetAllLines();
    }

    for (const auto &id : expressIds) {
        try {
            // ── geometry extraction (no GIL needed) ────────────────
            py::list verts_list;
            py::list faces_list;
            bool has_geometry = false;

            {
                std::lock_guard<std::mutex> lock(g_mutex);
                auto *geomProc = g_manager.GetGeometryProcessor(mid);

                webifc::geometry::IfcFlatMesh mesh = geomProc->GetFlatMesh(id);

                if (!mesh.geometries.empty()) {
                    // We may have multiple placed geometries per element —
                    // merge them, applying each placement transform.
                    uint32_t vertOffset = 0;

                    for (auto &placed : mesh.geometries) {
                        auto &geom = geomProc->GetGeometry(
                            placed.geometryExpressID
                        );

                        // Populate fvertexData from vertexData.
                        geom.GetVertexData();

                        uint32_t numFloats = geom.GetVertexDataSize();
                        uint32_t numIdx    = geom.GetIndexDataSize();
                        if (numFloats == 0 || numIdx == 0) continue;

                        const float *vptr =
                            reinterpret_cast<const float *>(
                                geom.GetVertexData()
                            );
                        const uint32_t *iptr =
                            reinterpret_cast<const uint32_t *>(
                                geom.GetIndexData()
                            );

                        uint32_t numVerts = numFloats / VFORMAT;

                        // The placement transform (column-major glm::dmat4).
                        const auto &T = placed.transformation;

                        // Extract position-only, transformed.
                        for (uint32_t v = 0; v < numVerts; ++v) {
                            double px = static_cast<double>(
                                vptr[v * VFORMAT + 0]
                            );
                            double py_val = static_cast<double>(
                                vptr[v * VFORMAT + 1]
                            );
                            double pz = static_cast<double>(
                                vptr[v * VFORMAT + 2]
                            );

                            // glm dmat4 is column-major:
                            //   T[col][row]
                            double wx = T[0][0]*px + T[1][0]*py_val
                                      + T[2][0]*pz + T[3][0];
                            double wy = T[0][1]*px + T[1][1]*py_val
                                      + T[2][1]*pz + T[3][1];
                            double wz = T[0][2]*px + T[1][2]*py_val
                                      + T[2][2]*pz + T[3][2];

                            verts_list.append(wx);
                            verts_list.append(wy);
                            verts_list.append(wz);
                        }

                        // Re-index faces with the current vertex offset.
                        for (uint32_t f = 0; f < numIdx; ++f) {
                            faces_list.append(
                                static_cast<int>(iptr[f] + vertOffset)
                            );
                        }

                        vertOffset += numVerts;
                        has_geometry = true;
                    }
                }

                // Free cached geometry — matches WASM StreamMeshes pattern.
                geomProc->Clear();
            }

            // ── invoke Python callback (GIL held) ──────────────────
            if (has_geometry) {
                callback(static_cast<int>(id), verts_list, faces_list);
            }

        } catch (const py::error_already_set &) {
            // Python exception in callback — propagate immediately.
            throw;
        } catch (const std::exception &ex) {
            std::cerr << "[webifc] Error processing expressID "
                      << id << ": " << ex.what() << "\n";
            // Continue to next element — never abort the full loop.
        } catch (...) {
            std::cerr << "[webifc] Unknown error processing expressID "
                      << id << "\n";
        }
    }
}

// ── get_all_lines ──────────────────────────────────────────────────────────

static std::vector<int> get_all_lines(int modelID)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    uint32_t mid = static_cast<uint32_t>(modelID);
    if (!g_manager.IsModelOpen(mid))
        throw std::runtime_error("Model not open");

    auto lines = g_manager.GetIfcLoader(mid)->GetAllLines();
    return std::vector<int>(lines.begin(), lines.end());
}

// ── get_line_type ──────────────────────────────────────────────────────────

static std::string get_line_type(int modelID, int expressID)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    uint32_t mid = static_cast<uint32_t>(modelID);
    uint32_t eid = static_cast<uint32_t>(expressID);

    if (!g_manager.IsModelOpen(mid))
        return "";

    try {
        uint32_t typeCode = g_manager.GetIfcLoader(mid)->GetLineType(eid);
        return g_manager.GetSchemaManager().IfcTypeCodeToType(typeCode);
    } catch (...) {
        return "";
    }
}

// ── get_string_argument ────────────────────────────────────────────────────

static std::string get_string_argument(int modelID, int expressID,
                                       int argIndex)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    uint32_t mid = static_cast<uint32_t>(modelID);
    uint32_t eid = static_cast<uint32_t>(expressID);
    uint32_t aidx = static_cast<uint32_t>(argIndex);

    if (!g_manager.IsModelOpen(mid))
        return "";

    try {
        auto *loader = g_manager.GetIfcLoader(mid);
        loader->MoveToArgumentOffset(eid, aidx);
        return loader->GetDecodedStringArgument();
    } catch (...) {
        return "";
    }
}

// ── close_model ────────────────────────────────────────────────────────────

static void close_model(int modelID)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    uint32_t mid = static_cast<uint32_t>(modelID);
    if (g_manager.IsModelOpen(mid))
        g_manager.CloseModel(mid);
}

// ── pybind11 module definition ─────────────────────────────────────────────

PYBIND11_MODULE(webifc, m)
{
    m.doc() = "Python bindings for the web-ifc geometry engine";

    m.def("open_model", &open_model, py::arg("path"),
          "Open an IFC file from a file path, returns integer modelID.");

    m.def("stream_all_meshes", &stream_all_meshes,
          py::arg("model_id"), py::arg("callback"),
          "Stream all meshes — calls callback(express_id, verts, faces) "
          "for each element with geometry. verts is flat "
          "[x,y,z,x,y,z,...], faces is flat [i,j,k,i,j,k,...].");

    m.def("get_all_lines", &get_all_lines, py::arg("model_id"),
          "Get all express IDs present in the model.");

    m.def("get_line_type", &get_line_type,
          py::arg("model_id"), py::arg("express_id"),
          "Get the IFC type string for an express ID, e.g. 'IFCWALL'.");

    m.def("get_string_argument", &get_string_argument,
          py::arg("model_id"), py::arg("express_id"),
          py::arg("arg_index"),
          "Get string attribute from an element by argument index.");

    m.def("close_model", &close_model, py::arg("model_id"),
          "Close a model and free memory.");
}
