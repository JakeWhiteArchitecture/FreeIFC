#!/usr/bin/env python3
"""FreeIFC — Lightweight desktop IFC viewer.

Usage:
    python3 freeifc.py [model.ifc]

Dependencies:
    pip install ifcopenshell PyQt6 trimesh numpy
    IfcConvert must be available on PATH (included with ifcopenshell).
"""

import sys
import os
import math
import shutil
import subprocess
import pickle
import numpy as np

import ifcopenshell

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QFileDialog, QScrollArea,
    QSizePolicy, QFrame, QProgressBar, QTreeWidget, QTreeWidgetItem,
    QMenu, QSlider, QColorDialog, QGroupBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QPoint, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor

import vtkmodules.vtkInteractionStyle  # noqa: F401 — registers styles
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 — registers OpenGL backend
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer, vtkCellPicker,
)
from vtkmodules.vtkRenderingLOD import vtkLODActor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ── IFC type colour / opacity map ────────────────────────────────────────────

TYPE_COLOURS = {
    "IfcWall":                (0.78, 0.78, 0.76),
    "IfcWallStandardCase":    (0.78, 0.78, 0.76),
    "IfcSlab":                (0.65, 0.65, 0.63),
    "IfcRoof":                (0.55, 0.42, 0.35),
    "IfcColumn":              (0.60, 0.60, 0.60),
    "IfcBeam":                (0.60, 0.60, 0.60),
    "IfcWindow":              (0.55, 0.72, 0.82),
    "IfcDoor":                (0.62, 0.48, 0.30),
    "IfcStair":               (0.65, 0.65, 0.63),
    "IfcStairFlight":         (0.65, 0.65, 0.63),
    "IfcRailing":             (0.50, 0.50, 0.48),
    "IfcPlate":               (0.55, 0.72, 0.82),
    "IfcCurtainWall":         (0.55, 0.72, 0.82),
    "IfcMember":              (0.58, 0.55, 0.48),
    "IfcFurnishingElement":   (0.70, 0.55, 0.40),
    "IfcCovering":            (0.82, 0.82, 0.80),
    "IfcSpace":               (0.70, 0.78, 0.88),
    "IfcOpeningElement":      (0.85, 0.85, 0.85),
    "IfcProxy":               (0.75, 0.70, 0.55),
    "IfcBuildingElementProxy":(0.70, 0.70, 0.72),
    "IfcFooting":             (0.55, 0.55, 0.52),
    "IfcPile":                (0.55, 0.55, 0.52),
}

TYPE_OPACITIES = {
    "IfcWindow":       0.35,
    "IfcCurtainWall":  0.50,
    "IfcSpace":        0.08,
}

DEFAULT_COLOUR  = (0.70, 0.70, 0.72)
DEFAULT_OPACITY = 1.0


def _colour_for(ifc_type: str):
    return TYPE_COLOURS.get(ifc_type, DEFAULT_COLOUR)


def _opacity_for(ifc_type: str):
    return TYPE_OPACITIES.get(ifc_type, DEFAULT_OPACITY)


# ── Loader thread — IfcConvert (fast) or ifcopenshell fallback + cache ───────

class LoaderThread(QThread):
    status_update = pyqtSignal(str)
    load_complete = pyqtSignal(dict)
    load_error    = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        cache_path = self.path + ".freeifc"
        ifc_mtime = os.path.getmtime(self.path)

        # ── Try cache ────────────────────────────────────────────────
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("mtime", 0) >= ifc_mtime:
                    self.status_update.emit("Loading cached geometry…")
                    elements = cache["elements"]
                    model = ifcopenshell.open(self.path)
                    for guid in list(elements.keys()):
                        try:
                            el = model.by_guid(guid)
                            elements[guid]["props"] = self._build_props(el)
                        except Exception:
                            elements[guid]["props"] = {
                                "Type": elements[guid].get("type", ""),
                                "GlobalId": guid,
                            }
                    hierarchy = self._build_hierarchy(model)
                    self.load_complete.emit({"elements": elements, "hierarchy": hierarchy})
                    return
            except Exception:
                pass

        # ── Try IfcConvert (fast C++ path) ───────────────────────────
        ifcconvert = shutil.which("IfcConvert") or shutil.which("ifcconvert")
        if ifcconvert:
            elements = self._load_via_ifcconvert(ifcconvert)
        else:
            elements = None

        # ── Fallback: ifcopenshell.geom.iterator ─────────────────────
        if elements is None:
            self.status_update.emit("Tessellating geometry (Python fallback)…")
            elements = self._load_via_iterator()

        if elements is None:
            return  # error already emitted

        if not elements:
            self.load_error.emit("No geometry found in IFC file.")
            return

        # ── Load IFC metadata ────────────────────────────────────────
        self.status_update.emit("Reading IFC metadata…")
        try:
            model = ifcopenshell.open(self.path)
        except Exception as exc:
            self.load_error.emit(f"Failed to read IFC metadata:\n{exc}")
            return

        for guid in list(elements.keys()):
            try:
                el = model.by_guid(guid)
                elements[guid]["type"] = el.is_a()
                elements[guid]["props"] = self._build_props(el)
            except Exception:
                elements[guid].setdefault("props", {"Type": elements[guid].get("type", ""), "GlobalId": guid})

        hierarchy = self._build_hierarchy(model)

        # ── Save cache ───────────────────────────────────────────────
        self.status_update.emit("Saving cache…")
        try:
            cache_data = {
                "mtime": ifc_mtime,
                "elements": {
                    guid: {"verts": e["verts"], "faces": e["faces"], "type": e.get("type", "")}
                    for guid, e in elements.items()
                },
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=5)
        except Exception:
            pass

        self.load_complete.emit({"elements": elements, "hierarchy": hierarchy})

    def _load_via_ifcconvert(self, ifcconvert: str) -> dict | None:
        """Fast path: IfcConvert → GLB → trimesh."""
        try:
            import trimesh
        except ImportError:
            return None  # trimesh not available, use fallback

        glb_path = self.path + ".tmp.glb"
        self.status_update.emit("Converting IFC geometry (IfcConvert)…")
        try:
            ncpu = os.cpu_count() or 4
            result = subprocess.run(
                [ifcconvert, "--use-element-guids", "-j", str(ncpu),
                 self.path, glb_path],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                return None  # fall back to iterator
        except Exception:
            return None

        self.status_update.emit("Loading GLB geometry…")
        try:
            scene = trimesh.load(glb_path, process=False)
        except Exception:
            return None
        finally:
            try:
                os.remove(glb_path)
            except Exception:
                pass

        self.status_update.emit("Processing geometry…")
        elements = {}

        if isinstance(scene, trimesh.Scene):
            for node_name in scene.graph.nodes_geometry:
                try:
                    T, geom_name = scene.graph[node_name]
                    mesh = scene.geometry[geom_name]
                    verts = np.array(mesh.vertices, dtype=np.float64)
                    if not np.allclose(T, np.eye(4)):
                        verts = (T[:3, :3] @ verts.T).T + T[:3, 3]
                    faces = np.array(mesh.faces, dtype=np.int64)
                    if len(verts) > 0 and len(faces) > 0:
                        elements[node_name] = {"verts": verts, "faces": faces, "type": ""}
                except Exception:
                    continue
        elif hasattr(scene, "vertices"):
            elements["unknown"] = {
                "verts": np.array(scene.vertices, dtype=np.float64),
                "faces": np.array(scene.faces, dtype=np.int64),
                "type": "",
            }

        return elements if elements else None

    def _load_via_iterator(self) -> dict | None:
        """Fallback: ifcopenshell.geom.iterator (slower but always works)."""
        import ifcopenshell.geom

        try:
            model = ifcopenshell.open(self.path)
        except Exception as exc:
            self.load_error.emit(f"Failed to open IFC file:\n{exc}")
            return None

        settings = ifcopenshell.geom.settings()
        try:
            settings.set("use-world-coords", True)
            settings.set("weld-vertices", True)
        except Exception:
            try:
                settings.set(settings.USE_WORLD_COORDS, True)
                settings.set(settings.WELD_VERTICES, True)
            except Exception:
                pass

        try:
            iterator = ifcopenshell.geom.iterator(settings, model, multiprocessing=True)
        except Exception:
            iterator = ifcopenshell.geom.iterator(settings, model)

        try:
            if not iterator.initialize():
                self.load_error.emit("No geometry found in IFC file.")
                return None
        except Exception as exc:
            self.load_error.emit(f"Geometry iterator failed:\n{exc}")
            return None

        elements = {}
        n_done = 0
        n_total = len(model.by_type("IfcProduct"))

        while True:
            try:
                shape = iterator.get()
                verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
                faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
                if len(verts) > 0 and len(faces) > 0:
                    elements[shape.guid] = {"verts": verts, "faces": faces, "type": ""}
            except Exception:
                pass

            n_done += 1
            if n_done % 200 == 0:
                self.status_update.emit(f"Tessellating… {n_done} / {n_total}")

            try:
                if not iterator.next():
                    break
            except Exception:
                break

        return elements

    def _build_props(self, element):
        props = {
            "Type": element.is_a(),
            "Name": getattr(element, "Name", None) or "",
            "GlobalId": element.GlobalId,
        }
        for attr in ("Description", "ObjectType", "Tag"):
            val = getattr(element, attr, None)
            if val:
                props[attr] = val
        return props

    def _build_hierarchy(self, model):
        def get_children(parent):
            children = []
            for rel in getattr(parent, "IsDecomposedBy", []):
                for child in rel.RelatedObjects:
                    children.append(child)
            return children

        def get_contained_guids(spatial_element):
            guids = []
            for rel in getattr(spatial_element, "ContainsElements", []):
                for el in rel.RelatedElements:
                    guids.append(el.GlobalId)
            return guids

        def build_node(element):
            name = getattr(element, "Name", None) or element.is_a()
            node = {
                "label": f"{name} [{element.is_a()}]",
                "type": element.is_a(),
                "guids": get_contained_guids(element),
                "children": [],
            }
            for child in get_children(element):
                node["children"].append(build_node(child))
            return node

        projects = model.by_type("IfcProject")
        if projects:
            root = build_node(projects[0])
        else:
            root = {"label": "(No Project)", "type": "", "guids": [], "children": []}

        return root


# ── Ground plane grid ────────────────────────────────────────────────────────

def _make_grid_actor(size=100.0, spacing=1.0):
    from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
    points = vtkPoints()
    lines = vtkCellArray()
    colours = vtkUnsignedCharArray()
    colours.SetNumberOfComponents(4)  # RGBA
    colours.SetName("Colours")
    n = int(size / spacing)
    half = n * spacing / 2.0
    idx = 0
    base_r, base_g, base_b = 46, 48, 54  # grid line colour
    fade_radius = half * 0.85  # fade starts at 85% of half-extent
    for i in range(n + 1):
        x = -half + i * spacing
        # Horizontal line: (x, -half) to (x, half)
        points.InsertNextPoint(x, -half, 0)
        points.InsertNextPoint(x,  half, 0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(idx)
        lines.InsertCellPoint(idx + 1)
        # Fade: alpha based on distance of each endpoint from origin
        for py in (-half, half):
            dist = math.sqrt(x * x + py * py)
            t = max(0.0, min(1.0, (dist - fade_radius) / (half - fade_radius)))
            alpha = int(255 * (1.0 - t * t))  # quadratic falloff, starts late
            colours.InsertNextTuple4(base_r, base_g, base_b, max(0, alpha))
        idx += 2
        # Vertical line: (-half, x) to (half, x)
        points.InsertNextPoint(-half, x, 0)
        points.InsertNextPoint( half, x, 0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(idx)
        lines.InsertCellPoint(idx + 1)
        for px in (-half, half):
            dist = math.sqrt(px * px + x * x)
            t = max(0.0, min(1.0, (dist - fade_radius) / (half - fade_radius)))
            alpha = int(255 * (1.0 - t * t))
            colours.InsertNextTuple4(base_r, base_g, base_b, max(0, alpha))
        idx += 2
    grid = vtkPolyData()
    grid.SetPoints(points)
    grid.SetLines(lines)
    grid.GetPointData().SetScalars(colours)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(grid)
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToDirectScalars()
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(1.0)
    actor.SetPickable(False)
    return actor


# ── Colour swatch button ────────────────────────────────────────────────────

class ColourSwatchButton(QPushButton):
    """Small button that shows a colour and opens a picker on click."""
    colour_changed = pyqtSignal(QColor)

    def __init__(self, initial: QColor, parent=None):
        super().__init__(parent)
        self._colour = initial
        self.setFixedSize(28, 28)
        self._update_style()
        self.clicked.connect(self._pick)

    def _update_style(self):
        self.setStyleSheet(
            f"background: {self._colour.name()}; border: 1px solid #555; "
            f"min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px; padding: 0;"
        )

    def colour(self) -> QColor:
        return self._colour

    def set_colour(self, c: QColor):
        self._colour = c
        self._update_style()

    def _pick(self):
        c = QColorDialog.getColor(self._colour, self, "Pick Colour")
        if c.isValid():
            self._colour = c
            self._update_style()
            self.colour_changed.emit(c)


# ── Main window ──────────────────────────────────────────────────────────────

class FreeIFCWindow(QMainWindow):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.setWindowTitle("FreeIFC")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        # Background settings
        self._bg_centre = QColor(128, 128, 140)  # lighter centre
        self._bg_edge   = QColor(255, 255, 255)  # brighter edge
        self._bg_intensity = 100                   # 0–200 scale percent

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_clicked)

        view_menu = menu_bar.addMenu("View")
        self._edge_action = view_menu.addAction("Show Edges")
        self._edge_action.setCheckable(True)
        self._edge_action.setChecked(True)
        self._edge_action.triggered.connect(self._on_toggle_edges)
        self._lod_action = view_menu.addAction("Enable LOD")
        self._lod_action.setCheckable(True)
        self._lod_action.setChecked(False)
        self._lod_action.triggered.connect(self._on_toggle_lod)
        reset_action = view_menu.addAction("Reset Visibility")
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self._on_reset_visibility)

        # State — merged architecture
        self._elements: dict[str, dict] = {}       # guid → {verts, faces, type, props}
        self._type_groups: dict[str, dict] = {}     # ifc_type → {guids, actor, edge_actor, cell_guids}
        self._hidden_guids: set[str] = set()
        self._selected_guid: str | None = None
        self._selection_outline: vtkActor | None = None
        self._loader: LoaderThread | None = None
        self._loading = False
        self._edges_visible = True
        self._lod_enabled = False
        self._lod_cloud_points = 5000000
        self._orbiting = False
        self._panning = False
        self._last_mouse_pos = (0, 0)
        self._right_press_pos = (0, 0)
        self._edge_build_timer = None
        self._edge_build_queue = []

        # ── Layout ───────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Side panel (left side)
        side = QWidget()
        side.setFixedWidth(320)
        side.setStyleSheet("""
            QWidget { background: #1e1f23; color: #d4d4d4; font-family: monospace; }
            QPushButton {
                background: #2d2e33; border: 1px solid #3a3b40;
                padding: 8px 12px; color: #e0e0e0; font-size: 13px;
            }
            QPushButton:hover { background: #3a3b40; }
            QPushButton:disabled { color: #666; }
            QLabel { font-size: 12px; }
            QTreeWidget {
                background: #1e1f23; color: #d4d4d4; border: none;
                font-size: 12px; font-family: monospace;
            }
            QTreeWidget::item { padding: 2px 0; }
            QTreeWidget::item:selected { background: #2d2e33; }
            QTreeWidget::branch { background: #1e1f23; }
            QHeaderView::section {
                background: #1e1f23; color: #888; border: none;
                font-size: 11px; font-weight: bold; padding: 4px;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #3a3b40; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px; height: 12px; margin: -4px 0;
                background: #888; border-radius: 6px;
            }
            QGroupBox {
                border: 1px solid #3a3b40; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
                font-size: 11px; color: #888;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)

        # Open button
        self._open_btn = QPushButton("Open IFC…")
        self._open_btn.clicked.connect(self._on_open_clicked)
        side_layout.addWidget(self._open_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m elements")
        self._progress.hide()
        side_layout.addWidget(self._progress)

        # Progress label
        self._progress_label = QLabel("")
        self._progress_label.hide()
        side_layout.addWidget(self._progress_label)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #3a3b40;")
        side_layout.addWidget(sep1)

        # Properties header
        props_header = QLabel("PROPERTIES")
        props_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
        side_layout.addWidget(props_header)

        self._props_label = QLabel("No element selected.")
        self._props_label.setWordWrap(True)
        self._props_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._props_label.setStyleSheet("font-size: 12px; padding: 4px 0;")
        side_layout.addWidget(self._props_label)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #3a3b40;")
        side_layout.addWidget(sep2)

        # Model tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("MODEL")
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        self._tree.itemChanged.connect(self._on_tree_item_changed)
        side_layout.addWidget(self._tree, stretch=1)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("color: #3a3b40;")
        side_layout.addWidget(sep3)

        # Background controls
        bg_group = QGroupBox("BACKGROUND")
        bg_lay = QVBoxLayout(bg_group)
        bg_lay.setSpacing(6)

        # Edge colour (outer)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Edge"))
        self._edge_swatch = ColourSwatchButton(self._bg_edge)
        self._edge_swatch.colour_changed.connect(self._on_bg_edge_changed)
        row1.addWidget(self._edge_swatch)
        row1.addStretch()
        # Centre colour (inner)
        row1.addWidget(QLabel("Centre"))
        self._centre_swatch = ColourSwatchButton(self._bg_centre)
        self._centre_swatch.colour_changed.connect(self._on_bg_centre_changed)
        row1.addWidget(self._centre_swatch)
        row1.addStretch()
        bg_lay.addLayout(row1)

        # Intensity slider
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Glow"))
        self._intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self._intensity_slider.setRange(0, 200)
        self._intensity_slider.setValue(self._bg_intensity)
        self._intensity_slider.valueChanged.connect(self._on_bg_intensity_changed)
        row2.addWidget(self._intensity_slider)
        self._intensity_label = QLabel(f"{self._bg_intensity}%")
        self._intensity_label.setFixedWidth(36)
        row2.addWidget(self._intensity_label)
        bg_lay.addLayout(row2)

        side_layout.addWidget(bg_group)

        # LOD controls
        lod_group = QGroupBox("LOD")
        lod_lay = QVBoxLayout(lod_group)
        lod_lay.setSpacing(6)
        self._lod_checkbox = QCheckBox("Enable LOD")
        self._lod_checkbox.setChecked(False)
        self._lod_checkbox.stateChanged.connect(self._on_lod_checkbox_changed)
        lod_lay.addWidget(self._lod_checkbox)
        lod_row = QHBoxLayout()
        lod_row.addWidget(QLabel("Detail"))
        self._lod_slider = QSlider(Qt.Orientation.Horizontal)
        self._lod_slider.setRange(100000, 10000000)
        self._lod_slider.setValue(self._lod_cloud_points)
        self._lod_slider.setSingleStep(500000)
        self._lod_slider.valueChanged.connect(self._on_lod_slider_changed)
        self._lod_slider.setEnabled(False)
        lod_row.addWidget(self._lod_slider)
        self._lod_label = QLabel("5M")
        self._lod_label.setFixedWidth(36)
        lod_row.addWidget(self._lod_label)
        lod_lay.addLayout(lod_row)
        side_layout.addWidget(lod_group)

        main_layout.addWidget(side)

        # VTK widget (right of sidebar)
        self._vtk_widget = QVTKRenderWindowInteractor(central)
        main_layout.addWidget(self._vtk_widget, stretch=1)

        # ── VTK setup ────────────────────────────────────────────────
        renwin = self._vtk_widget.GetRenderWindow()

        # Layer 0 — scene
        self.renderer = vtkRenderer()
        self.renderer.SetLayer(0)
        self._apply_background()
        # Depth peeling for transparency
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.1)
        renwin.SetAlphaBitPlanes(True)
        renwin.SetMultiSamples(0)  # must be 0 for depth peeling
        self.renderer.SetUseFXAA(True)  # anti-aliasing for lines
        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(self.renderer)

        # Layer 1 — outline overlay (renders on top of all scene geometry)
        self.overlay = vtkRenderer()
        self.overlay.SetLayer(1)
        self.overlay.InteractiveOff()
        self.overlay.SetUseFXAA(True)
        # Preserve colour from layer 0 but clear depth so outlines always on top
        self.overlay.SetPreserveColorBuffer(True)
        self.overlay.SetPreserveDepthBuffer(False)
        renwin.AddRenderer(self.overlay)
        self.overlay.SetActiveCamera(self.renderer.GetActiveCamera())

        # Set initial camera to isometric angle
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(10, -10, 8)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)

        # Ground grid
        self._grid_actor = _make_grid_actor()
        self.renderer.AddActor(self._grid_actor)

        # XYZ axes widget (like Blender) — top right
        axes = vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.05)
        axes.SetConeRadius(0.15)
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        # Red X, Green Y, Blue Z — Blender convention
        axes.GetXAxisShaftProperty().SetColor(0.9, 0.2, 0.25)
        axes.GetXAxisTipProperty().SetColor(0.9, 0.2, 0.25)
        axes.GetYAxisShaftProperty().SetColor(0.55, 0.75, 0.15)
        axes.GetYAxisTipProperty().SetColor(0.55, 0.75, 0.15)
        axes.GetZAxisShaftProperty().SetColor(0.3, 0.55, 0.95)
        axes.GetZAxisTipProperty().SetColor(0.3, 0.55, 0.95)
        for prop in (axes.GetXAxisCaptionActor2D().GetCaptionTextProperty(),
                     axes.GetYAxisCaptionActor2D().GetCaptionTextProperty(),
                     axes.GetZAxisCaptionActor2D().GetCaptionTextProperty()):
            prop.SetFontSize(14)
            prop.SetBold(True)
            prop.SetItalic(False)
            prop.SetShadow(False)
            prop.SetColor(0.85, 0.85, 0.85)
        self._orientation_widget = vtkOrientationMarkerWidget()
        self._orientation_widget.SetOrientationMarker(axes)
        self._orientation_widget.SetViewport(0.85, 0.85, 1.0, 1.0)

        # Neutered trackball style — override all button handlers to no-ops
        style = vtkInteractorStyleTrackballCamera()
        style.AddObserver("LeftButtonPressEvent", lambda o, e: None)
        style.AddObserver("LeftButtonReleaseEvent", lambda o, e: None)
        style.AddObserver("RightButtonPressEvent", lambda o, e: None)
        style.AddObserver("RightButtonReleaseEvent", lambda o, e: None)
        style.AddObserver("MiddleButtonPressEvent", lambda o, e: None)
        style.AddObserver("MiddleButtonReleaseEvent", lambda o, e: None)
        style.AddObserver("MouseWheelForwardEvent", lambda o, e: None)
        style.AddObserver("MouseWheelBackwardEvent", lambda o, e: None)
        style.AddObserver("MouseMoveEvent", lambda o, e: None)

        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        iren.SetInteractorStyle(style)

        # Our own interaction handlers
        iren.AddObserver("LeftButtonPressEvent", self._on_left_press)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        iren.AddObserver("MiddleButtonPressEvent", self._on_middle_press)
        iren.AddObserver("MiddleButtonReleaseEvent", self._on_middle_release)
        iren.AddObserver("MouseMoveEvent", self._on_mouse_move)
        iren.AddObserver("RightButtonPressEvent", self._on_right_press)
        iren.AddObserver("RightButtonReleaseEvent", self._on_right_release)
        iren.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
        iren.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)

        # Picker — cell-level for merged actor picking
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.002)

        # Start interactor
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()

        # Enable orientation widget after interactor is started
        self._orientation_widget.SetInteractor(iren)
        self._orientation_widget.EnabledOn()
        self._orientation_widget.InteractiveOff()

        # Load initial file if provided
        if initial_file and os.path.isfile(initial_file):
            self._load_file(initial_file)

    # ── Background ───────────────────────────────────────────────────────

    def _apply_background(self):
        """Apply gradient background with intensity scaling."""
        scale = self._bg_intensity / 100.0
        cr = min(1.0, self._bg_centre.redF() * scale)
        cg = min(1.0, self._bg_centre.greenF() * scale)
        cb = min(1.0, self._bg_centre.blueF() * scale)
        er = min(1.0, self._bg_edge.redF() * scale)
        eg = min(1.0, self._bg_edge.greenF() * scale)
        eb = min(1.0, self._bg_edge.blueF() * scale)
        self.renderer.SetBackground(er, eg, eb)
        self.renderer.SetBackground2(cr, cg, cb)
        self.renderer.GradientBackgroundOn()
        try:
            self.renderer.SetGradientMode(
                vtkRenderer.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER
            )
        except Exception:
            pass

    def _on_bg_centre_changed(self, colour: QColor):
        self._bg_centre = colour
        self._apply_background()
        self._vtk_widget.GetRenderWindow().Render()

    def _on_bg_edge_changed(self, colour: QColor):
        self._bg_edge = colour
        self._apply_background()
        self._vtk_widget.GetRenderWindow().Render()

    def _on_bg_intensity_changed(self, value: int):
        self._bg_intensity = value
        self._intensity_label.setText(f"{value}%")
        self._apply_background()
        self._vtk_widget.GetRenderWindow().Render()

    # ── LOD controls ─────────────────────────────────────────────────────

    def _on_toggle_lod(self, checked: bool):
        self._lod_enabled = checked
        self._lod_checkbox.setChecked(checked)
        self._apply_lod()

    def _on_lod_checkbox_changed(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self._lod_enabled = enabled
        self._lod_action.setChecked(enabled)
        self._lod_slider.setEnabled(enabled)
        self._apply_lod()

    def _on_lod_slider_changed(self, value: int):
        self._lod_cloud_points = value
        if value >= 1000000:
            self._lod_label.setText(f"{value / 1000000:.0f}M")
        else:
            self._lod_label.setText(f"{value / 1000:.0f}K")
        self._apply_lod()

    def _apply_lod(self):
        pts = self._lod_cloud_points if self._lod_enabled else 999999999
        for group in self._type_groups.values():
            if group["actor"]:
                group["actor"].SetNumberOfCloudPoints(pts)
        self._vtk_widget.GetRenderWindow().Render()

    # ── Interaction handlers ─────────────────────────────────────────────

    def _on_left_press(self, obj, event):
        """Left-click: selection only."""
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        click_pos = iren.GetEventPosition()

        self._picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = self._picker.GetActor()
        cell_id = self._picker.GetCellId()

        if picked_actor is not None and picked_actor is not self._grid_actor and cell_id >= 0:
            guid = self._guid_from_pick(picked_actor, cell_id)
            if guid is not None:
                if guid == self._selected_guid:
                    self._deselect()
                else:
                    self._select(guid)
                self._vtk_widget.GetRenderWindow().Render()
                return

        # Clicked empty space — deselect
        if self._selected_guid is not None:
            self._deselect()
            self._vtk_widget.GetRenderWindow().Render()

    def _on_left_release(self, obj, event):
        pass

    def _on_middle_press(self, obj, event):
        """Middle-click: pan."""
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        self._panning = True
        self._last_mouse_pos = iren.GetEventPosition()

    def _on_middle_release(self, obj, event):
        self._panning = False

    def _on_right_press(self, obj, event):
        """Right-press: start orbit drag, track for click detection."""
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        self._orbiting = True
        pos = iren.GetEventPosition()
        self._last_mouse_pos = pos
        self._right_press_pos = pos

    def _on_right_release(self, obj, event):
        self._orbiting = False
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        pos = iren.GetEventPosition()
        dx = abs(pos[0] - self._right_press_pos[0])
        dy = abs(pos[1] - self._right_press_pos[1])
        if dx < 4 and dy < 4:
            # Right-click (no drag) — context menu
            self._picker.Pick(pos[0], pos[1], 0, self.renderer)
            picked_actor = self._picker.GetActor()
            cell_id = self._picker.GetCellId()
            if picked_actor is not None and picked_actor is not self._grid_actor and cell_id >= 0:
                guid = self._guid_from_pick(picked_actor, cell_id)
                if guid is not None:
                    self._show_hide_menu(guid, pos)

    def _on_mouse_move(self, obj, event):
        if not self._orbiting and not self._panning:
            return

        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        pos = iren.GetEventPosition()
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = pos

        cam = self.renderer.GetActiveCamera()

        if self._orbiting:
            cam.Azimuth(-dx * 0.5)
            cam.Elevation(-dy * 0.5)
            cam.SetViewUp(0, 0, 1)
            cam.OrthogonalizeViewUp()
        elif self._panning:
            renwin = self._vtk_widget.GetRenderWindow()
            size = renwin.GetSize()
            fp = cam.GetFocalPoint()
            pos3d = cam.GetPosition()
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos3d, fp)))
            view_angle = cam.GetViewAngle()
            pan_scale = 2.0 * dist * math.tan(math.radians(view_angle / 2.0)) / size[1]
            cam.OrthogonalizeViewUp()
            right = [0, 0, 0]
            up = list(cam.GetViewUp())
            view_dir = [fp[i] - pos3d[i] for i in range(3)]
            right[0] = view_dir[1] * up[2] - view_dir[2] * up[1]
            right[1] = view_dir[2] * up[0] - view_dir[0] * up[2]
            right[2] = view_dir[0] * up[1] - view_dir[1] * up[0]
            mag = math.sqrt(sum(r * r for r in right))
            if mag > 0:
                right = [r / mag for r in right]
            pan_x = -dx * pan_scale
            pan_y = -dy * pan_scale
            delta = [pan_x * right[i] + pan_y * up[i] for i in range(3)]
            cam.SetFocalPoint(*[fp[i] + delta[i] for i in range(3)])
            cam.SetPosition(*[pos3d[i] + delta[i] for i in range(3)])

        self.renderer.ResetCameraClippingRange()
        self._vtk_widget.GetRenderWindow().Render()

    def _on_wheel_forward(self, obj, event):
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(1.1)
        self.renderer.ResetCameraClippingRange()
        self._vtk_widget.GetRenderWindow().Render()

    def _on_wheel_backward(self, obj, event):
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(0.9)
        self.renderer.ResetCameraClippingRange()
        self._vtk_widget.GetRenderWindow().Render()

    # ── Picking helper ───────────────────────────────────────────────────

    def _guid_from_pick(self, actor, cell_id: int) -> str | None:
        """Map a picked actor + cell_id to the element's GUID."""
        for ifc_type, group in self._type_groups.items():
            if group["actor"] is actor:
                cell_guids = group.get("cell_guids", [])
                if 0 <= cell_id < len(cell_guids):
                    return cell_guids[cell_id]
                return None
        return None

    # ── Edge toggle ──────────────────────────────────────────────────────

    def _on_toggle_edges(self, checked: bool):
        self._edges_visible = checked
        for group in self._type_groups.values():
            if group["edge_actor"]:
                group["edge_actor"].SetVisibility(checked)
        self._vtk_widget.GetRenderWindow().Render()

    # ── Reset visibility ─────────────────────────────────────────────────

    def _on_reset_visibility(self):
        if not self._hidden_guids:
            return
        affected_types = {self._elements[g]["type"] for g in self._hidden_guids if g in self._elements}
        self._hidden_guids.clear()
        for ifc_type in affected_types:
            self._rebuild_type_actor(ifc_type)
        self._tree.blockSignals(True)
        self._reset_tree_checks(self._tree.invisibleRootItem())
        self._tree.blockSignals(False)
        self._vtk_widget.GetRenderWindow().Render()

    def _reset_tree_checks(self, item):
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, Qt.CheckState.Checked)
            self._reset_tree_checks(child)

    # ── Context menu ─────────────────────────────────────────────────────

    def _show_hide_menu(self, guid: str, click_pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2d2e33; color: #d4d4d4; border: 1px solid #3a3b40; }
            QMenu::item:selected { background: #3a3b40; }
        """)
        props = self._elements.get(guid, {}).get("props", {})
        name = props.get("Name", "") or guid[:8]
        hide_action = menu.addAction(f"Hide: {name}")
        hide_action.triggered.connect(lambda: self._hide_element(guid))

        vtk_size = self._vtk_widget.GetRenderWindow().GetSize()
        local_pos = self._vtk_widget.mapToGlobal(
            self._vtk_widget.rect().topLeft()
        )
        global_x = local_pos.x() + click_pos[0]
        global_y = local_pos.y() + (vtk_size[1] - click_pos[1])
        menu.popup(QPoint(global_x, global_y))

    def _hide_element(self, guid: str):
        self._hidden_guids.add(guid)
        if guid == self._selected_guid:
            self._deselect()
        ifc_type = self._elements.get(guid, {}).get("type", "")
        if ifc_type:
            self._rebuild_type_actor(ifc_type)
        self._vtk_widget.GetRenderWindow().Render()

    # ── Drag and drop ────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._loading:
            return
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.toLocalFile().lower().endswith(".ifc"):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        if self._loading:
            return
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".ifc"):
                self._load_file(path)
                return

    # ── File open ────────────────────────────────────────────────────────

    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open IFC File", "", "IFC Files (*.ifc);;All Files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        if self._loading:
            return

        self._clear_scene()

        self._loading = True
        self._open_btn.setEnabled(False)
        self.setAcceptDrops(False)
        self._progress.setMaximum(0)  # indeterminate
        self._progress.show()
        self._progress_label.setText("Starting…")
        self._progress_label.show()

        self.setWindowTitle(f"FreeIFC — {os.path.basename(path)}")

        self._loader = LoaderThread(path)
        self._loader.status_update.connect(self._on_status_update)
        self._loader.load_complete.connect(self._on_load_complete)
        self._loader.load_error.connect(self._on_load_error)
        self._loader.start()

    def _clear_scene(self):
        self._deselect()
        if self._edge_build_timer is not None:
            self._edge_build_timer.stop()
            self._edge_build_timer = None
        self._edge_build_queue = []
        for group in self._type_groups.values():
            if group["actor"]:
                self.renderer.RemoveActor(group["actor"])
            if group["edge_actor"]:
                self.renderer.RemoveActor(group["edge_actor"])
        self._type_groups.clear()
        self._elements.clear()
        self._hidden_guids.clear()
        self._tree.clear()
        self._vtk_widget.GetRenderWindow().Render()

    # ── Loader signals ───────────────────────────────────────────────────

    def _on_status_update(self, msg: str):
        self._progress_label.setText(msg)

    def _on_load_complete(self, data: dict):
        self._loading = False
        self._open_btn.setEnabled(True)
        self.setAcceptDrops(True)

        self._elements = data["elements"]
        hierarchy = data["hierarchy"]

        # Group elements by IFC type
        type_map: dict[str, list[str]] = {}
        for guid, elem in self._elements.items():
            ifc_type = elem.get("type", "Unknown")
            type_map.setdefault(ifc_type, []).append(guid)

        # Build merged actors per type
        self._progress_label.setText("Building scene…")
        for ifc_type, guids in type_map.items():
            self._type_groups[ifc_type] = {
                "guids": guids,
                "actor": None,
                "edge_actor": None,
                "cell_guids": [],
            }
            self._rebuild_type_actor(ifc_type)

        # Populate tree
        self._tree.blockSignals(True)
        self._tree.clear()
        self._populate_tree(None, hierarchy)
        self._tree.expandToDepth(1)
        self._tree.blockSignals(False)

        # Auto-zoom with isometric camera
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        fp = list(cam.GetFocalPoint())
        cam.SetPosition(fp[0] + 10, fp[1] - 10, fp[2] + 8)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        cam.Dolly(0.85)
        self.renderer.ResetCameraClippingRange()
        self._vtk_widget.GetRenderWindow().Render()

        # Build edges in background after scene is visible
        self._progress_label.setText("Building edges…")
        self._edge_build_queue = list(self._type_groups.keys())
        self._edge_build_timer = QTimer()
        self._edge_build_timer.setInterval(0)
        self._edge_build_timer.timeout.connect(self._build_edges_batch)
        self._edge_build_timer.start()

    def _on_load_error(self, msg: str):
        self._loading = False
        self._open_btn.setEnabled(True)
        self.setAcceptDrops(True)
        self._progress.hide()
        self._progress_label.setText(msg)
        self._progress_label.show()

    # ── Merged actor building ────────────────────────────────────────────

    def _rebuild_type_actor(self, ifc_type: str):
        """Rebuild the merged scene actor for one IFC type."""
        group = self._type_groups.get(ifc_type)
        if group is None:
            return

        # Remove old actor
        if group["actor"]:
            self.renderer.RemoveActor(group["actor"])
            group["actor"] = None

        # Concatenate visible elements
        all_verts = []
        all_faces = []
        cell_guids = []
        vert_offset = 0

        for guid in group["guids"]:
            if guid in self._hidden_guids:
                continue
            elem = self._elements.get(guid)
            if elem is None:
                continue
            v = elem["verts"]
            f = elem["faces"]
            all_verts.append(v)
            all_faces.append(f + vert_offset)
            cell_guids.extend([guid] * len(f))
            vert_offset += len(v)

        group["cell_guids"] = cell_guids

        if not all_verts:
            return

        merged_v = np.concatenate(all_verts)
        merged_f = np.concatenate(all_faces)

        polydata = _make_polydata(merged_v, merged_f)
        normals = vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        colour = _colour_for(ifc_type)
        opacity = _opacity_for(ifc_type)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        actor = vtkLODActor()
        actor.SetMapper(mapper)
        lod_pts = self._lod_cloud_points if self._lod_enabled else 999999999
        actor.SetNumberOfCloudPoints(lod_pts)
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetInterpolationToFlat()
        self.renderer.AddActor(actor)
        group["actor"] = actor

    def _rebuild_type_edges(self, ifc_type: str):
        """Rebuild the merged edge actor for one IFC type."""
        group = self._type_groups.get(ifc_type)
        if group is None:
            return

        if group["edge_actor"]:
            self.renderer.RemoveActor(group["edge_actor"])
            group["edge_actor"] = None

        if not self._edges_visible:
            return

        # Concatenate visible elements
        all_verts = []
        all_faces = []
        vert_offset = 0

        for guid in group["guids"]:
            if guid in self._hidden_guids:
                continue
            elem = self._elements.get(guid)
            if elem is None:
                continue
            v = elem["verts"]
            f = elem["faces"]
            all_verts.append(v)
            all_faces.append(f + vert_offset)
            vert_offset += len(v)

        if not all_verts:
            return

        merged_v = np.concatenate(all_verts)
        merged_f = np.concatenate(all_faces)

        polydata = _make_polydata(merged_v, merged_f)
        normals = vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        fe = vtkFeatureEdges()
        fe.SetInputData(normals.GetOutput())
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOn()
        fe.SetFeatureAngle(30)
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ColoringOff()
        fe.Update()

        edge_mapper = vtkPolyDataMapper()
        edge_mapper.SetInputConnection(fe.GetOutputPort())
        edge_actor = vtkActor()
        edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetColor(0.12, 0.12, 0.14)
        edge_actor.GetProperty().SetLineWidth(1.0)
        edge_actor.GetProperty().SetAmbient(1.0)
        edge_actor.GetProperty().SetDiffuse(0.0)
        edge_actor.GetProperty().LightingOff()
        edge_actor.SetPickable(False)
        self.renderer.AddActor(edge_actor)
        group["edge_actor"] = edge_actor

    def _build_edges_batch(self):
        """Build edge actors one type per tick."""
        if not self._edge_build_queue:
            self._edge_build_timer.stop()
            self._edge_build_timer = None
            self._progress.hide()
            self._progress_label.hide()
            self._vtk_widget.GetRenderWindow().Render()
            return

        ifc_type = self._edge_build_queue.pop()
        self._rebuild_type_edges(ifc_type)
        remaining = len(self._edge_build_queue)
        total = len(self._type_groups)
        self._progress_label.setText(f"Building edges… {total - remaining} / {total} types")
        if remaining % 3 == 0:
            self._vtk_widget.GetRenderWindow().Render()

    # ── Tree ─────────────────────────────────────────────────────────────

    def _populate_tree(self, parent_item, node: dict):
        if parent_item is None:
            item = QTreeWidgetItem(self._tree)
        else:
            item = QTreeWidgetItem(parent_item)

        item.setText(0, node["label"])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
        item.setCheckState(0, Qt.CheckState.Checked)
        guids = node.get("guids", [])
        item.setData(0, Qt.ItemDataRole.UserRole, guids)

        for child in node.get("children", []):
            self._populate_tree(item, child)

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int):
        visible = item.checkState(0) != Qt.CheckState.Unchecked
        affected_types = set()
        self._collect_visibility_changes(item, visible, affected_types)
        for ifc_type in affected_types:
            self._rebuild_type_actor(ifc_type)
            self._rebuild_type_edges(ifc_type)
        self._vtk_widget.GetRenderWindow().Render()

    def _collect_visibility_changes(self, item, visible: bool, affected_types: set):
        guids = item.data(0, Qt.ItemDataRole.UserRole) or []
        for guid in guids:
            if guid not in self._elements:
                continue
            ifc_type = self._elements[guid]["type"]
            if visible:
                self._hidden_guids.discard(guid)
            else:
                self._hidden_guids.add(guid)
                if guid == self._selected_guid:
                    self._deselect()
            affected_types.add(ifc_type)

        for i in range(item.childCount()):
            child = item.child(i)
            child_visible = child.checkState(0) != Qt.CheckState.Unchecked
            self._collect_visibility_changes(
                child, child_visible if visible else False, affected_types
            )

    # ── Selection ────────────────────────────────────────────────────────

    def _select(self, guid: str):
        self._deselect()
        self._selected_guid = guid

        elem = self._elements.get(guid)
        if elem is None:
            return

        # Create outline actor from element's individual geometry
        polydata = _make_polydata(elem["verts"], elem["faces"])
        normals = vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        fe = vtkFeatureEdges()
        fe.SetInputData(normals.GetOutput())
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOn()
        fe.SetFeatureAngle(30)
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ColoringOff()
        fe.Update()

        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(fe.GetOutputPort())
        outline_actor = vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetLineWidth(2.5)
        outline_actor.GetProperty().SetColor(0.31, 0.76, 0.97)
        outline_actor.GetProperty().SetAmbient(1.0)
        outline_actor.GetProperty().SetDiffuse(0.0)
        outline_actor.GetProperty().LightingOff()
        outline_actor.SetPickable(False)
        self.overlay.AddActor(outline_actor)
        self._selection_outline = outline_actor

        # Update properties panel
        props = elem.get("props", {})
        lines = []
        for key in ("Type", "Name", "GlobalId", "Description", "ObjectType", "Tag"):
            val = props.get(key)
            if val:
                lines.append(f"<b>{key}:</b> {val}")
        self._props_label.setText("<br>".join(lines))

    def _deselect(self):
        if self._selected_guid is None:
            return
        if self._selection_outline:
            self.overlay.RemoveActor(self._selection_outline)
            self._selection_outline = None
        self._selected_guid = None
        self._props_label.setText("No element selected.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_polydata(verts: np.ndarray, faces: np.ndarray) -> vtkPolyData:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    # Points — zero-copy from numpy
    verts_c = np.ascontiguousarray(verts, dtype=np.float64)
    vtk_pts_data = numpy_to_vtk(verts_c, deep=True)
    pts = vtkPoints()
    pts.SetData(vtk_pts_data)

    # Cells — build connectivity array: [3, i0, i1, i2, 3, i0, i1, i2, ...]
    n_tri = len(faces)
    conn = np.empty((n_tri, 4), dtype=np.int64)
    conn[:, 0] = 3
    conn[:, 1:] = faces
    vtk_cells_data = numpy_to_vtkIdTypeArray(conn.ravel(), deep=True)
    cells = vtkCellArray()
    cells.SetCells(n_tri, vtk_cells_data)

    pd = vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FreeIFC")

    initial_file = sys.argv[1] if len(sys.argv) > 1 else None
    window = FreeIFCWindow(initial_file=initial_file)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
