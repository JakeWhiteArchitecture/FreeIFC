#!/usr/bin/env python3
"""FreeIFC — Lightweight desktop IFC viewer.

Usage:
    python3 freeifc.py [model.ifc]

Dependencies:
    pip install ifcopenshell PyQt6 pyvista
"""

import sys
import os
import numpy as np

import ifcopenshell
import ifcopenshell.geom

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QFileDialog, QScrollArea,
    QSizePolicy, QFrame, QProgressBar, QTreeWidget, QTreeWidgetItem,
    QMenu,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

import vtkmodules.vtkInteractionStyle  # noqa: F401 — registers styles
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 — registers OpenGL backend
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer,
)
from vtkmodules.vtkRenderingLOD import vtkLODActor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain
from vtkmodules.vtkRenderingCore import vtkCellPicker
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor
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


# ── Main window ──────────────────────────────────────────────────────────────

class FreeIFCWindow(QMainWindow):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.setWindowTitle("FreeIFC")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_clicked)

        view_menu = menu_bar.addMenu("View")
        self._edge_action = view_menu.addAction("Show Edges")
        self._edge_action.setCheckable(True)
        self._edge_action.setChecked(False)
        self._edge_action.triggered.connect(self._on_toggle_edges)
        reset_action = view_menu.addAction("Reset Visibility")
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self._on_reset_visibility)

        # State
        self._actors: dict[str, vtkLODActor] = {}       # guid → scene actor
        self._edge_actors: dict[str, vtkActor] = {}      # guid → feature-edge actor
        self._outlines: dict[str, vtkActor] = {}          # guid → outline actor
        self._props: dict[str, dict] = {}                 # guid → property dict
        self._hidden_guids: set[str] = set()              # manually hidden via right-click
        self._selected_guid: str | None = None
        self._loader: LoaderThread | None = None
        self._loading = False
        self._edges_visible = False

        # ── Layout ───────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # VTK widget
        self._vtk_widget = QVTKRenderWindowInteractor(central)
        main_layout.addWidget(self._vtk_widget, stretch=1)

        # Side panel
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

        main_layout.addWidget(side)

        # ── VTK setup ────────────────────────────────────────────────────
        renwin = self._vtk_widget.GetRenderWindow()

        # Layer 0 — scene
        self.renderer = vtkRenderer()
        # Radial gradient background — subtle centre glow, dark edges
        self.renderer.SetBackground(0.04, 0.045, 0.06)   # edge colour (dark)
        self.renderer.SetBackground2(0.12, 0.13, 0.16)    # centre colour (lighter)
        self.renderer.GradientBackgroundOn()
        try:
            # VTK 9.2+ supports radial gradient mode
            self.renderer.SetGradientMode(
                vtkRenderer.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER
            )
        except Exception:
            pass  # Fall back to linear gradient on older VTK
        self.renderer.SetLayer(0)
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

        # Set initial camera to isometric angle to avoid parallel view-up warning
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(10, -10, 8)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 0, 1)

        # Ground grid
        self._grid_actor = _make_grid_actor()
        self.renderer.AddActor(self._grid_actor)

        # Navigation cube
        cube = vtkAnnotatedCubeActor()
        cube.SetXPlusFaceText("E")
        cube.SetXMinusFaceText("W")
        cube.SetYPlusFaceText("N")
        cube.SetYMinusFaceText("S")
        cube.SetZPlusFaceText("Top")
        cube.SetZMinusFaceText("Bot")
        cube.GetTextEdgesProperty().SetColor(0.18, 0.18, 0.18)
        cube.GetTextEdgesProperty().SetLineWidth(1)
        cube.GetCubeProperty().SetColor(0.24, 0.25, 0.28)
        for face_prop in (
            cube.GetXPlusFaceProperty(), cube.GetXMinusFaceProperty(),
            cube.GetYPlusFaceProperty(), cube.GetYMinusFaceProperty(),
            cube.GetZPlusFaceProperty(), cube.GetZMinusFaceProperty(),
        ):
            face_prop.SetColor(0.24, 0.25, 0.28)

        self._orientation_widget = vtkOrientationMarkerWidget()
        self._orientation_widget.SetOrientationMarker(cube)
        self._orientation_widget.SetViewport(0.85, 0.0, 1.0, 0.15)

        # Interaction style
        style = vtkInteractorStyleTerrain()

        def _wheel_forward(obj, event):
            c = self.renderer.GetActiveCamera()
            c.Dolly(1.1)
            self.renderer.ResetCameraClippingRange()
            self._vtk_widget.GetRenderWindow().Render()

        def _wheel_backward(obj, event):
            c = self.renderer.GetActiveCamera()
            c.Dolly(0.9)
            self.renderer.ResetCameraClippingRange()
            self._vtk_widget.GetRenderWindow().Render()

        style.AddObserver("MouseWheelForwardEvent", _wheel_forward)
        style.AddObserver("MouseWheelBackwardEvent", _wheel_backward)

        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        iren.SetInteractorStyle(style)

        # Picker
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)

        # Left-click for selection
        iren.AddObserver("LeftButtonPressEvent", self._on_left_click)
        # Right-click for context menu
        iren.AddObserver("RightButtonPressEvent", self._on_right_click)

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

    # ── Edge toggle (feature edges only, no co-planar) ───────────────────

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
        # Restore all actors
        for guid, actor in self._actors.items():
            actor.SetVisibility(True)
        for guid, actor in self._edge_actors.items():
            actor.SetVisibility(self._edges_visible)
        # Reset tree checkboxes
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

        # Find guid
        guid = None
        for g, a in self._actors.items():
            if a is picked_actor:
                guid = g
                break

        if guid is None:
            iren.GetInteractorStyle().OnRightButtonDown()
            return

        # Show context menu at cursor position
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background: #2d2e33; color: #d4d4d4; border: 1px solid #3a3b40; }
            QMenu::item:selected { background: #3a3b40; }
        """)
        name = self._props.get(guid, {}).get("Name", "") or guid[:8]
        hide_action = menu.addAction(f"Hide: {name}")
        hide_action.triggered.connect(lambda: self._hide_element(guid))

        # Map VTK screen coords to Qt global coords
        vtk_size = self._vtk_widget.GetRenderWindow().GetSize()
        local_pos = self._vtk_widget.mapToGlobal(
            self._vtk_widget.rect().topLeft()
        )
        # VTK y is bottom-up, Qt y is top-down
        global_x = local_pos.x() + click_pos[0]
        global_y = local_pos.y() + (vtk_size[1] - click_pos[1])

        from PyQt6.QtCore import QPoint
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

        # Scene actor (LOD) — flat shading, no light source
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        actor = vtkLODActor()
        actor.SetMapper(mapper)
        actor.SetNumberOfCloudPoints(5000000)
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.GetProperty().LightingOff()
        actor.GetProperty().SetInterpolationToFlat()
        self.renderer.AddActor(actor)
        self._actors[guid] = actor

        # Feature-edge actor for "Show Edges" toggle — only hard edges
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

        # Build tree
        self._tree.blockSignals(True)
        self._tree.clear()
        self._populate_tree(None, hierarchy)
        self._tree.expandToDepth(1)
        self._tree.blockSignals(False)

        # Auto-zoom: set isometric camera and fit to model
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        # Position camera at isometric angle to avoid view-up parallel warning
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
                continue  # Skip manually hidden elements
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

    def _on_left_click(self, obj, event):
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        click_pos = iren.GetEventPosition()
        self._picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = self._picker.GetActor()

        if picked_actor is None or picked_actor is self._grid_actor:
            self._deselect()
            self._vtk_widget.GetRenderWindow().Render()
            iren.GetInteractorStyle().OnLeftButtonDown()
            return

        guid = None
        for g, a in self._actors.items():
            if a is picked_actor:
                guid = g
                break

        if guid is None:
            iren.GetInteractorStyle().OnLeftButtonDown()
            return

        if guid == self._selected_guid:
            self._deselect()
        else:
            self._select(guid)

        self._vtk_widget.GetRenderWindow().Render()
        iren.GetInteractorStyle().OnLeftButtonDown()

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
