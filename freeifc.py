#!/usr/bin/env python3
"""FreeIFC — Lightweight desktop IFC viewer.

Usage:
    python3 freeifc.py [model.ifc]

Dependencies:
    pip install ifcopenshell PyQt6 pyvista
"""

import sys
import os
import math
import numpy as np

import ifcopenshell
import ifcopenshell.geom

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QFileDialog, QScrollArea,
    QSizePolicy, QFrame, QProgressBar, QTreeWidget, QTreeWidgetItem,
    QMenu, QSlider, QColorDialog, QGroupBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QPoint
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor

import vtkmodules.vtkInteractionStyle  # noqa: F401 — registers styles
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 — registers OpenGL backend
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer,
)
from vtkmodules.vtkRenderingLOD import vtkLODActor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ── IFC type colour / opacity map ────────────────────────────────────────────

TYPE_COLOURS = {
    "IfcWall":                (0.78, 0.78, 0.76),
    "IfcWallStandardCase":    (0.78, 0.78, 0.76),
    "IfcSlab":                (0.60, 0.62, 0.65),
    "IfcRoof":                (0.55, 0.45, 0.38),
    "IfcColumn":              (0.72, 0.68, 0.58),
    "IfcBeam":                (0.72, 0.68, 0.58),
    "IfcDoor":                (0.85, 0.65, 0.40),
    "IfcWindow":              (0.55, 0.80, 0.92),
    "IfcCurtainWall":         (0.55, 0.80, 0.92),
    "IfcStair":               (0.70, 0.60, 0.48),
    "IfcStairFlight":         (0.70, 0.60, 0.48),
    "IfcRailing":             (0.50, 0.50, 0.55),
    "IfcSpace":               (0.40, 0.65, 0.90),
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


# ── Background loader thread ────────────────────────────────────────────────

class LoaderThread(QThread):
    element_ready = pyqtSignal(str, object, object, dict)
    load_progress = pyqtSignal(int, int)
    load_complete = pyqtSignal(dict)
    load_error    = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            model = ifcopenshell.open(self.path)
        except Exception as exc:
            self.load_error.emit(f"Failed to open IFC file:\n{exc}")
            return

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
            iterator = ifcopenshell.geom.iterator(
                settings, model, multiprocessing=True,
            )
        except Exception:
            iterator = ifcopenshell.geom.iterator(settings, model)

        try:
            if not iterator.initialize():
                self.load_error.emit("No geometry found in IFC file.")
                return
        except Exception as exc:
            self.load_error.emit(f"Geometry iterator failed to initialize:\n{exc}")
            return

        n_done = 0
        all_products = model.by_type("IfcProduct")
        n_total = len(all_products)

        while True:
            try:
                shape = iterator.get()
                element = model.by_guid(shape.guid)
                ifc_type = element.is_a()

                verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
                faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)

                if len(verts) > 0 and len(faces) > 0:
                    props = {
                        "Type": ifc_type,
                        "Name": getattr(element, "Name", None) or "",
                        "GlobalId": shape.guid,
                    }
                    for attr in ("Description", "ObjectType", "Tag"):
                        val = getattr(element, attr, None)
                        if val:
                            props[attr] = val

                    self.element_ready.emit(shape.guid, verts, faces, props)
            except Exception:
                pass

            n_done += 1
            self.load_progress.emit(n_done, n_total)

            try:
                if not iterator.next():
                    break
            except Exception:
                break

        hierarchy = self._build_hierarchy(model)
        self.load_complete.emit(hierarchy)

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
    points = vtkPoints()
    lines = vtkCellArray()
    n = int(size / spacing)
    half = n * spacing / 2.0
    idx = 0
    for i in range(n + 1):
        x = -half + i * spacing
        points.InsertNextPoint(x, -half, 0)
        points.InsertNextPoint(x,  half, 0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(idx)
        lines.InsertCellPoint(idx + 1)
        idx += 2
        points.InsertNextPoint(-half, x, 0)
        points.InsertNextPoint( half, x, 0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(idx)
        lines.InsertCellPoint(idx + 1)
        idx += 2
    grid = vtkPolyData()
    grid.SetPoints(points)
    grid.SetLines(lines)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(grid)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.18, 0.19, 0.21)
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
        self._bg_centre = QColor(30, 31, 40)    # lighter centre
        self._bg_edge   = QColor(10, 12, 15)    # darker edge
        self._bg_intensity = 100                  # 0–200 scale percent

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

        # State
        self._actors: dict[str, vtkLODActor] = {}
        self._edge_actors: dict[str, vtkActor] = {}
        self._outlines: dict[str, vtkActor] = {}
        self._props: dict[str, dict] = {}
        self._hidden_guids: set[str] = set()
        self._selected_guid: str | None = None
        self._loader: LoaderThread | None = None
        self._loading = False
        self._edges_visible = True
        self._lod_enabled = False
        self._lod_cloud_points = 5000000
        self._orbiting = False
        self._panning = False
        self._last_mouse_pos = (0, 0)

        # ── Layout ───────────────────────────────────────────────────────
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

        # ── VTK setup ────────────────────────────────────────────────────
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
        renwin.SetMultiSamples(0)
        renwin.SetNumberOfLayers(2)
        renwin.AddRenderer(self.renderer)

        # Layer 1 — outline overlay
        self.overlay = vtkRenderer()
        self.overlay.SetLayer(1)
        self.overlay.SetErase(False)
        self.overlay.InteractiveOff()
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
        axes.SetTipLength(0.2)
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

        # Use trackball camera but we fully override left-button via observers
        # to prevent the base style's rotate mode from getting stuck
        bare_style = vtkInteractorStyleTrackballCamera()

        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        iren.SetInteractorStyle(bare_style)

        # Left-button: click to select, drag to orbit
        iren.AddObserver("LeftButtonPressEvent", self._on_left_press)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        # Middle-button: pan
        iren.AddObserver("MiddleButtonPressEvent", self._on_middle_press)
        iren.AddObserver("MiddleButtonReleaseEvent", self._on_middle_release)
        # Mouse move: orbit or pan depending on state
        iren.AddObserver("MouseMoveEvent", self._on_mouse_move)
        # Right-click for context menu
        iren.AddObserver("RightButtonPressEvent", self._on_right_click)
        # Scroll zoom
        iren.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
        iren.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)

        # Picker
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)

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
        # Centre colour (Background in VTK = centre for radial)
        cr = min(1.0, self._bg_centre.redF() * scale)
        cg = min(1.0, self._bg_centre.greenF() * scale)
        cb = min(1.0, self._bg_centre.blueF() * scale)
        # Edge colour (Background2 in VTK = edge for radial)
        er = min(1.0, self._bg_edge.redF() * scale)
        eg = min(1.0, self._bg_edge.greenF() * scale)
        eb = min(1.0, self._bg_edge.blueF() * scale)
        self.renderer.SetBackground(er, eg, eb)    # VTK: Background = edge
        self.renderer.SetBackground2(cr, cg, cb)    # VTK: Background2 = centre
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
        for actor in self._actors.values():
            actor.SetNumberOfCloudPoints(pts)
        self._vtk_widget.GetRenderWindow().Render()

    # ── Custom orbit (Z-axis azimuth) ────────────────────────────────────

    def _on_left_press(self, obj, event):
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        click_pos = iren.GetEventPosition()

        # Pick for selection
        self._picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = self._picker.GetActor()

        if picked_actor is not None and picked_actor is not self._grid_actor:
            guid = None
            for g, a in self._actors.items():
                if a is picked_actor:
                    guid = g
                    break
            if guid is not None:
                if guid == self._selected_guid:
                    self._deselect()
                else:
                    self._select(guid)
                self._vtk_widget.GetRenderWindow().Render()
                # Don't start orbit when clicking on an element
                return
        else:
            if self._selected_guid is not None:
                self._deselect()
                self._vtk_widget.GetRenderWindow().Render()

        # Start orbit
        self._orbiting = True
        self._last_mouse_pos = click_pos

    def _on_left_release(self, obj, event):
        self._orbiting = False
        # Ensure the base style exits any mode it might have entered
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        iren.GetInteractorStyle().OnLeftButtonUp()

    def _on_middle_press(self, obj, event):
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        self._panning = True
        self._last_mouse_pos = iren.GetEventPosition()

    def _on_middle_release(self, obj, event):
        self._panning = False
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        iren.GetInteractorStyle().OnMiddleButtonUp()

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
            # Horizontal drag → azimuth (rotate around Z axis)
            cam.Azimuth(-dx * 0.5)
            # Vertical drag → elevation (tilt up/down)
            cam.Elevation(-dy * 0.5)
            # Lock up-vector to Z to prevent flipping
            cam.SetViewUp(0, 0, 1)
            cam.OrthogonalizeViewUp()
        elif self._panning:
            # Pan: translate camera and focal point
            renwin = self._vtk_widget.GetRenderWindow()
            size = renwin.GetSize()
            fp = cam.GetFocalPoint()
            pos3d = cam.GetPosition()
            # Calculate pan amount based on distance from camera to focal point
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos3d, fp)))
            view_angle = cam.GetViewAngle()
            pan_scale = 2.0 * dist * math.tan(math.radians(view_angle / 2.0)) / size[1]
            # Get camera axes
            cam.OrthogonalizeViewUp()
            right = [0, 0, 0]
            up = list(cam.GetViewUp())
            view_dir = [fp[i] - pos3d[i] for i in range(3)]
            # Cross product: right = view_dir x up
            right[0] = view_dir[1] * up[2] - view_dir[2] * up[1]
            right[1] = view_dir[2] * up[0] - view_dir[0] * up[2]
            right[2] = view_dir[0] * up[1] - view_dir[1] * up[0]
            # Normalize
            mag = math.sqrt(sum(r * r for r in right))
            if mag > 0:
                right = [r / mag for r in right]
            # Pan
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

    # ── Edge toggle ──────────────────────────────────────────────────────

    def _on_toggle_edges(self, checked: bool):
        self._edges_visible = checked
        for guid, actor in self._edge_actors.items():
            if checked and guid not in self._hidden_guids:
                actor.SetVisibility(True)
            else:
                actor.SetVisibility(False)
        self._vtk_widget.GetRenderWindow().Render()

    # ── Reset visibility ─────────────────────────────────────────────────

    def _on_reset_visibility(self):
        self._hidden_guids.clear()
        for guid, actor in self._actors.items():
            actor.SetVisibility(True)
        for guid, actor in self._edge_actors.items():
            actor.SetVisibility(self._edges_visible)
        self._tree.blockSignals(True)
        self._reset_tree_checks(self._tree.invisibleRootItem())
        self._tree.blockSignals(False)
        self._vtk_widget.GetRenderWindow().Render()

    def _reset_tree_checks(self, item):
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, Qt.CheckState.Checked)
            self._reset_tree_checks(child)

    # ── Right-click context menu ─────────────────────────────────────────

    def _on_right_click(self, obj, event):
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        click_pos = iren.GetEventPosition()
        self._picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = self._picker.GetActor()

        if picked_actor is None or picked_actor is self._grid_actor:
            iren.GetInteractorStyle().OnRightButtonDown()
            return

        guid = None
        for g, a in self._actors.items():
            if a is picked_actor:
                guid = g
                break

        if guid is None:
            iren.GetInteractorStyle().OnRightButtonDown()
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2d2e33; color: #d4d4d4; border: 1px solid #3a3b40; }
            QMenu::item:selected { background: #3a3b40; }
        """)
        name = self._props.get(guid, {}).get("Name", "") or guid[:8]
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
        actor = self._actors.get(guid)
        if actor:
            actor.SetVisibility(False)
        edge_actor = self._edge_actors.get(guid)
        if edge_actor:
            edge_actor.SetVisibility(False)
        outline = self._outlines.get(guid)
        if outline:
            outline.SetVisibility(False)
        if guid == self._selected_guid:
            self._deselect()
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
        self._progress.setValue(0)
        self._progress.show()
        self._progress_label.setText("Loading…")
        self._progress_label.show()

        self.setWindowTitle(f"FreeIFC — {os.path.basename(path)}")

        self._loader = LoaderThread(path)
        self._loader.element_ready.connect(self._on_element_ready)
        self._loader.load_progress.connect(self._on_load_progress)
        self._loader.load_complete.connect(self._on_load_complete)
        self._loader.load_error.connect(self._on_load_error)
        self._loader.start()

    def _clear_scene(self):
        self._deselect()
        for actor in self._actors.values():
            self.renderer.RemoveActor(actor)
        for actor in self._edge_actors.values():
            self.renderer.RemoveActor(actor)
        for actor in self._outlines.values():
            self.overlay.RemoveActor(actor)
        self._actors.clear()
        self._edge_actors.clear()
        self._outlines.clear()
        self._props.clear()
        self._hidden_guids.clear()
        self._tree.clear()
        self._vtk_widget.GetRenderWindow().Render()

    # ── Loader signals ───────────────────────────────────────────────────

    def _on_element_ready(self, guid: str, verts: np.ndarray, faces: np.ndarray, props: dict):
        ifc_type = props.get("Type", "")
        colour = _colour_for(ifc_type)
        opacity = _opacity_for(ifc_type)

        polydata = _make_polydata(verts, faces)

        normals = vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        # Scene actor — flat shading, no light source
        # Use vtkLODActor but disable LOD by default (huge cloud points = never switch)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        actor = vtkLODActor()
        actor.SetMapper(mapper)
        if self._lod_enabled:
            actor.SetNumberOfCloudPoints(self._lod_cloud_points)
        else:
            actor.SetNumberOfCloudPoints(999999999)  # effectively disables LOD
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetInterpolationToFlat()
        self.renderer.AddActor(actor)
        self._actors[guid] = actor

        # Feature-edge actor — visible by default (lines where forms overlap)
        fe_edge = vtkFeatureEdges()
        fe_edge.SetInputData(normals.GetOutput())
        fe_edge.BoundaryEdgesOn()
        fe_edge.FeatureEdgesOn()
        fe_edge.SetFeatureAngle(30)
        fe_edge.ManifoldEdgesOff()
        fe_edge.NonManifoldEdgesOff()
        fe_edge.ColoringOff()
        fe_edge.Update()
        edge_mapper = vtkPolyDataMapper()
        edge_mapper.SetInputConnection(fe_edge.GetOutputPort())
        edge_actor = vtkActor()
        edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetColor(0.12, 0.12, 0.14)
        edge_actor.GetProperty().SetLineWidth(1.0)
        edge_actor.GetProperty().SetAmbient(1.0)
        edge_actor.GetProperty().SetDiffuse(0.0)
        edge_actor.GetProperty().LightingOff()
        edge_actor.SetPickable(False)
        edge_actor.SetVisibility(self._edges_visible)
        self.renderer.AddActor(edge_actor)
        self._edge_actors[guid] = edge_actor

        # Outline actor — hard edges for selection highlight
        fe_outline = vtkFeatureEdges()
        fe_outline.SetInputData(normals.GetOutput())
        fe_outline.BoundaryEdgesOn()
        fe_outline.FeatureEdgesOn()
        fe_outline.SetFeatureAngle(30)
        fe_outline.ManifoldEdgesOff()
        fe_outline.NonManifoldEdgesOff()
        fe_outline.ColoringOff()
        fe_outline.Update()
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(fe_outline.GetOutputPort())
        outline_actor = vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetLineWidth(2.5)
        outline_actor.GetProperty().SetColor(0.31, 0.76, 0.97)
        outline_actor.GetProperty().SetAmbient(1.0)
        outline_actor.GetProperty().SetDiffuse(0.0)
        outline_actor.GetProperty().LightingOff()
        outline_actor.SetVisibility(False)
        outline_actor.SetPickable(False)
        self.overlay.AddActor(outline_actor)
        self._outlines[guid] = outline_actor

        self._props[guid] = props

        if len(self._actors) % 20 == 0:
            self._vtk_widget.GetRenderWindow().Render()

    def _on_load_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._progress_label.setText(f"Loaded {done} / {total} elements")

    def _on_load_complete(self, hierarchy: dict):
        self._loading = False
        self._open_btn.setEnabled(True)
        self.setAcceptDrops(True)
        self._progress.hide()
        self._progress_label.hide()

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
        self._set_subtree_visibility(item, visible)
        self._vtk_widget.GetRenderWindow().Render()

    def _set_subtree_visibility(self, item: QTreeWidgetItem, visible: bool):
        guids = item.data(0, Qt.ItemDataRole.UserRole) or []
        for guid in guids:
            if guid in self._hidden_guids:
                continue
            actor = self._actors.get(guid)
            if actor:
                actor.SetVisibility(visible)
            edge_actor = self._edge_actors.get(guid)
            if edge_actor:
                edge_actor.SetVisibility(visible and self._edges_visible)
            outline = self._outlines.get(guid)
            if outline and not visible:
                outline.SetVisibility(False)
            if not visible and guid == self._selected_guid:
                self._deselect()

        for i in range(item.childCount()):
            child = item.child(i)
            child_visible = child.checkState(0) != Qt.CheckState.Unchecked
            self._set_subtree_visibility(child, child_visible if visible else False)

    def _on_load_error(self, msg: str):
        self._loading = False
        self._open_btn.setEnabled(True)
        self.setAcceptDrops(True)
        self._progress.hide()
        self._progress_label.setText(msg)
        self._progress_label.show()

    # ── Selection ────────────────────────────────────────────────────────

    def _select(self, guid: str):
        self._deselect()

        self._selected_guid = guid
        actor = self._actors.get(guid)
        if actor:
            prop = actor.GetProperty()
            new_opacity = min(1.0, prop.GetOpacity() + 0.18)
            prop.SetOpacity(new_opacity)

        outline = self._outlines.get(guid)
        if outline:
            outline.SetVisibility(True)

        props = self._props.get(guid, {})
        lines = []
        for key in ("Type", "Name", "GlobalId", "Description", "ObjectType", "Tag"):
            val = props.get(key)
            if val:
                lines.append(f"<b>{key}:</b> {val}")
        self._props_label.setText("<br>".join(lines))

    def _deselect(self):
        if self._selected_guid is None:
            return
        guid = self._selected_guid
        actor = self._actors.get(guid)
        if actor:
            ifc_type = self._props.get(guid, {}).get("Type", "")
            actor.GetProperty().SetOpacity(_opacity_for(ifc_type))

        outline = self._outlines.get(guid)
        if outline:
            outline.SetVisibility(False)

        self._selected_guid = None
        self._props_label.setText("No element selected.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_polydata(verts: np.ndarray, faces: np.ndarray) -> vtkPolyData:
    pts = vtkPoints()
    pts.SetNumberOfPoints(len(verts))
    for i, (x, y, z) in enumerate(verts):
        pts.SetPoint(i, x, y, z)

    cells = vtkCellArray()
    for tri in faces:
        cells.InsertNextCell(3)
        for idx in tri:
            cells.InsertCellPoint(int(idx))

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
