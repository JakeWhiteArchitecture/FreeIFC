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
    QSizePolicy, QFrame, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

import vtkmodules.vtkInteractionStyle  # noqa: F401 — registers styles
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 — registers OpenGL backend
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkPolyDataMapper, vtkRenderer,
)
from vtkmodules.vtkRenderingLOD import vtkLODActor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkCellPicker

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
    """Tessellates IFC geometry on a background thread.

    Signals emitted back to the main thread:
        element_ready  — one element tessellated (guid, verts, faces, props)
        load_progress  — (n_done, n_total)
        load_complete  — {storey_name: [guid, …], …}
        load_error     — str error message
    """

    element_ready = pyqtSignal(str, object, object, dict)   # guid, verts, faces, props
    load_progress = pyqtSignal(int, int)                     # done, total
    load_complete = pyqtSignal(dict)                         # storey_map
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

        # Geometry iterator settings
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        settings.set("weld-vertices", True)

        iterator = ifcopenshell.geom.iterator(
            settings, model, multiprocessing=True,
        )

        if not iterator.initialize():
            self.load_error.emit("No geometry found in IFC file.")
            return

        n_done = 0
        # Count total products with representations for progress
        all_products = model.by_type("IfcProduct")
        n_total = len(all_products)

        while True:
            shape = iterator.get()
            element = model.by_guid(shape.guid)
            ifc_type = element.is_a()

            try:
                verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
                faces = np.array(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
            except Exception:
                n_done += 1
                self.load_progress.emit(n_done, n_total)
                if not iterator.next():
                    break
                continue

            if len(verts) == 0 or len(faces) == 0:
                n_done += 1
                self.load_progress.emit(n_done, n_total)
                if not iterator.next():
                    break
                continue

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

            n_done += 1
            self.load_progress.emit(n_done, n_total)

            if not iterator.next():
                break

        # Build storey map
        storey_map: dict[str, list[str]] = {}
        assigned_guids: set[str] = set()

        for storey in model.by_type("IfcBuildingStorey"):
            name = storey.Name or f"Storey {storey.GlobalId[:8]}"
            guids: list[str] = []
            for rel in getattr(storey, "ContainsElements", []):
                for el in rel.RelatedElements:
                    guids.append(el.GlobalId)
                    assigned_guids.add(el.GlobalId)
            storey_map[name] = guids

        # Collect unassigned elements
        unassigned: list[str] = []
        for product in all_products:
            if product.GlobalId not in assigned_guids:
                # Only include elements that actually had geometry
                unassigned.append(product.GlobalId)
        if unassigned:
            storey_map["(Unassigned)"] = unassigned

        self.load_complete.emit(storey_map)


# ── Ground plane grid ────────────────────────────────────────────────────────

def _make_grid_actor(size=100.0, spacing=1.0):
    """Create a flat grid of lines on the XY plane (Z=0)."""
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

        # State
        self._actors: dict[str, vtkLODActor] = {}       # guid → scene actor
        self._outlines: dict[str, vtkActor] = {}         # guid → outline actor
        self._props: dict[str, dict] = {}                # guid → property dict
        self._storey_map: dict[str, list[str]] = {}
        self._selected_guid: str | None = None
        self._loader: LoaderThread | None = None
        self._loading = False

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
            QCheckBox { font-size: 12px; spacing: 6px; }
            QCheckBox::indicator { width: 14px; height: 14px; }
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

        # Storeys header
        storeys_header = QLabel("STOREYS")
        storeys_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
        side_layout.addWidget(storeys_header)

        # Scrollable storey list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        self._storey_container = QWidget()
        self._storey_layout = QVBoxLayout(self._storey_container)
        self._storey_layout.setContentsMargins(0, 0, 0, 0)
        self._storey_layout.setSpacing(4)
        self._storey_layout.addStretch()
        scroll.setWidget(self._storey_container)
        side_layout.addWidget(scroll, stretch=1)

        main_layout.addWidget(side)

        # ── VTK setup ────────────────────────────────────────────────────
        renwin = self._vtk_widget.GetRenderWindow()

        # Layer 0 — scene
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.08, 0.09, 0.11)
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
        # Share camera
        self.overlay.SetActiveCamera(self.renderer.GetActiveCamera())

        # Lights — key + fill
        self.renderer.RemoveAllLights()
        self.renderer.CreateLight()
        from vtkmodules.vtkRenderingCore import vtkLight
        key = vtkLight()
        key.SetLightTypeToSceneLight()
        key.SetPosition(5, 5, 10)
        key.SetFocalPoint(0, 0, 0)
        key.SetIntensity(0.9)
        key.SetColor(1.0, 0.98, 0.95)
        self.renderer.AddLight(key)
        fill = vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(-3, -6, 4)
        fill.SetFocalPoint(0, 0, 0)
        fill.SetIntensity(0.45)
        fill.SetColor(0.85, 0.90, 1.0)
        self.renderer.AddLight(fill)

        # Ground grid
        self._grid_actor = _make_grid_actor()
        self.renderer.AddActor(self._grid_actor)

        # Interaction style
        style = vtkInteractorStyleTrackballCamera()
        self._vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        # Picker
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.005)

        # Connect left-click for selection
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        iren.AddObserver("LeftButtonPressEvent", self._on_left_click)

        # Start interactor
        self._vtk_widget.Initialize()
        self._vtk_widget.Start()

        # Load initial file if provided
        if initial_file and os.path.isfile(initial_file):
            self._load_file(initial_file)

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

        # Clear current scene
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
        for actor in self._outlines.values():
            self.overlay.RemoveActor(actor)
        self._actors.clear()
        self._outlines.clear()
        self._props.clear()
        self._storey_map.clear()
        # Clear storey checkboxes
        while self._storey_layout.count() > 1:
            item = self._storey_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._vtk_widget.GetRenderWindow().Render()

    # ── Loader signals ───────────────────────────────────────────────────

    def _on_element_ready(self, guid: str, verts: np.ndarray, faces: np.ndarray, props: dict):
        ifc_type = props.get("Type", "")
        colour = _colour_for(ifc_type)
        opacity = _opacity_for(ifc_type)

        # Build vtkPolyData
        polydata = _make_polydata(verts, faces)

        # Normals
        normals = vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.Update()

        # Scene actor (LOD)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        actor = vtkLODActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToPhong()
        self.renderer.AddActor(actor)
        self._actors[guid] = actor

        # Outline actor (standard, hidden by default)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputData(normals.GetOutput())
        outline_actor = vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetRepresentationToWireframe()
        outline_actor.GetProperty().SetLineWidth(2.0)
        outline_actor.GetProperty().SetColor(0.31, 0.76, 0.97)
        outline_actor.GetProperty().SetAmbient(1.0)
        outline_actor.GetProperty().SetDiffuse(0.0)
        outline_actor.GetProperty().LightingOff()
        outline_actor.SetVisibility(False)
        outline_actor.SetPickable(False)
        self.overlay.AddActor(outline_actor)
        self._outlines[guid] = outline_actor

        self._props[guid] = props

        # Render periodically (every few elements)
        if len(self._actors) % 20 == 0:
            self._vtk_widget.GetRenderWindow().Render()

    def _on_load_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._progress_label.setText(f"Loaded {done} / {total} elements")

    def _on_load_complete(self, storey_map: dict):
        self._storey_map = storey_map
        self._loading = False
        self._open_btn.setEnabled(True)
        self.setAcceptDrops(True)
        self._progress.hide()
        self._progress_label.hide()

        # Build storey checkboxes
        while self._storey_layout.count() > 1:
            item = self._storey_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for name in storey_map:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, n=name: self._on_storey_toggled(n, state))
            self._storey_layout.insertWidget(self._storey_layout.count() - 1, cb)

        # Reset camera
        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Dolly(0.85)
        self.renderer.ResetCameraClippingRange()
        self._vtk_widget.GetRenderWindow().Render()

    def _on_load_error(self, msg: str):
        self._loading = False
        self._open_btn.setEnabled(True)
        self.setAcceptDrops(True)
        self._progress.hide()
        self._progress_label.setText(msg)
        self._progress_label.show()

    # ── Storey visibility ────────────────────────────────────────────────

    def _on_storey_toggled(self, storey_name: str, state: int):
        visible = state == Qt.CheckState.Checked.value
        guids = self._storey_map.get(storey_name, [])
        for guid in guids:
            actor = self._actors.get(guid)
            if actor:
                actor.SetVisibility(visible)
            outline = self._outlines.get(guid)
            if outline:
                if not visible:
                    outline.SetVisibility(False)
        # If selected element is hidden, deselect
        if not visible and self._selected_guid in guids:
            self._deselect()
        self._vtk_widget.GetRenderWindow().Render()

    # ── Selection ────────────────────────────────────────────────────────

    def _on_left_click(self, obj, event):
        # Forward to default handler first
        iren = self._vtk_widget.GetRenderWindow().GetInteractor()
        click_pos = iren.GetEventPosition()
        self._picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        picked_actor = self._picker.GetActor()

        if picked_actor is None or picked_actor is self._grid_actor:
            # Clicked empty space or grid — deselect
            self._deselect()
            self._vtk_widget.GetRenderWindow().Render()
            iren.GetInteractorStyle().OnLeftButtonDown()
            return

        # Find which guid this actor belongs to
        guid = None
        for g, a in self._actors.items():
            if a is picked_actor:
                guid = g
                break

        if guid is None:
            iren.GetInteractorStyle().OnLeftButtonDown()
            return

        if guid == self._selected_guid:
            # Click same element — toggle off
            self._deselect()
        else:
            self._select(guid)

        self._vtk_widget.GetRenderWindow().Render()
        iren.GetInteractorStyle().OnLeftButtonDown()

    def _select(self, guid: str):
        # Deselect previous
        self._deselect()

        self._selected_guid = guid
        actor = self._actors.get(guid)
        if actor:
            prop = actor.GetProperty()
            new_opacity = min(1.0, prop.GetOpacity() + 0.18)
            prop.SetOpacity(new_opacity)
            prop.SetAmbient(0.4)

        outline = self._outlines.get(guid)
        if outline:
            outline.SetVisibility(True)

        # Update properties panel
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
            actor.GetProperty().SetAmbient(0.0)

        outline = self._outlines.get(guid)
        if outline:
            outline.SetVisibility(False)

        self._selected_guid = None
        self._props_label.setText("No element selected.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_polydata(verts: np.ndarray, faces: np.ndarray) -> vtkPolyData:
    """Convert numpy arrays to vtkPolyData."""
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
