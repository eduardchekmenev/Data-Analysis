# Spinsolve_DataAnalysis_gui.py — date-folder picker refactor (with session log + layout tweaks)
# Updates:
# 1) Default main path set to: D:\WSU\Raw Data\Spinsolve-1.4T_13C
# 2) Plot area given a minimum height; folder selection area is the flexible region.
# 3) Pop-up dialogs replaced by an in-window session log to the right of the folder list.
# 4) Integral shown in standard numeric format (not scientific) above the plot.

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QLabel, QPushButton, QDoubleSpinBox, QDial, QTextEdit,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

# ---------------------------
# Processing utilities
# ---------------------------

def exponential_apodization(fid, time, lb):
    return fid * np.exp(-lb * time)

def zero_fill(fid, target_length=65536):
    if len(fid) >= target_length:
        return fid
    return np.pad(fid, (0, target_length - len(fid)), mode='constant')

def apply_phase_correction(fid, angle_deg):
    return fid * np.exp(-1j * np.deg2rad(angle_deg))

def compute_fft(fid, dt):
    spectrum = np.fft.fftshift(np.fft.fft(fid))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fid), d=dt))
    return freqs, spectrum

def optimize_phase(fid, time_arr):
    dt = float(np.mean(np.diff(time_arr)))

    def objective(angle):
        corr = apply_phase_correction(fid, angle)
        _, spectrum = compute_fft(corr, dt)
        # Maximize power in real(spectrum) => minimize negative sum of squares
        return -np.sum(np.real(spectrum) ** 2)

    res = minimize_scalar(objective, bounds=(0.0, 10.0), method='bounded')
    return float(res.x)

def read_fid_csv(path):
    df = pd.read_csv(path, header=None)
    # Expect columns: time(ms), real, imag
    time = df.iloc[:, 0].astype(float).to_numpy() / 1000.0  # ms -> s
    real = df.iloc[:, 1].astype(float).to_numpy()
    imag = df.iloc[:, 2].astype(float).to_numpy()
    fid = real + 1j * imag
    return fid, time

def ppm_conversion(freqs):
    # User-provided conversion (kept identical):
    return -((freqs - (2500 - 639.08016399999997)) / 15.507665)

def process_and_plot(fid, time, lb, user_phase_offset, ppm_start, ppm_stop):
    fid_apod = exponential_apodization(fid, time, lb)
    opt_phase = optimize_phase(fid_apod, time)
    total_phase = opt_phase + user_phase_offset
    fid_corr = apply_phase_correction(fid_apod, total_phase)
    fid_zf = zero_fill(fid_corr)

    dt = float(time[1] - time[0])
    freqs, spectrum = compute_fft(fid_zf, dt)
    ppm = ppm_conversion(freqs)

    real = np.real(spectrum)
    # Simple baseline estimate from far frequencies
    baseline = np.mean(real[np.abs(freqs) > 200]) if np.any(np.abs(freqs) > 200) else 0.0
    real = real - baseline

    # Window selection in the conventional NMR orientation (high ppm on left)
    mask = (ppm > ppm_stop) & (ppm < ppm_start)
    ppm_vals = ppm[mask]
    real_vals = real[mask]

    # Sort by ascending ppm for plotting/integration
    order = np.argsort(ppm_vals)
    ppm_vals = ppm_vals[order]
    real_vals = real_vals[order]

    # Numeric integration
    integral = float(np.trapz(real_vals, ppm_vals)) if len(ppm_vals) > 1 else 0.0

    # Make figure
    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(ppm, real, lw=1)
    ax.set_xlabel("ppm")
    ax.set_ylabel("Real FFT (a.u.)")
    # Reverse ppm axis (high right to low left)
    ax.invert_xaxis()

    # Highlight integration window
    if len(ppm_vals) > 1:
        ax.fill_between(ppm_vals, real_vals, alpha=0.2)

    # Integral shown in regular numeric format (with thousands separator, 3 decimals)
    ax.set_title(f"Integral: {integral:,.3f} | Phase offset: {user_phase_offset:.2f}° (opt {opt_phase:.2f}°)")
    fig.tight_layout()
    return fig

# ---------------------------
# Main Window
# ---------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spinsolve Data Analysis — Date/Folder Picker")

        # State holders
        self.selected_folders = []  # names relative to base/date
        self.data = []              # list of (fid, time, folder, transient_num)
        self.phase_offsets = []     # per-trace manual offsets (deg)
        self.figs = []              # cache of figures per-trace
        self.current = 0            # index into self.data

        # ---------------- Layout ----------------
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top form
        form = QFormLayout()
        self.path_edit = QLineEdit(r"D:\WSU\Raw Data\Spinsolve-1.4T_13C")
        self.date_edit = QLineEdit("2025-06-11")

        self.set_date_btn = QPushButton("Set Date")
        self.set_date_btn.clicked.connect(self.on_set_date)

        top_row = QHBoxLayout()
        top_row.addWidget(self.path_edit)
        top_row.addWidget(self.date_edit)
        top_row.addWidget(self.set_date_btn)
        form.addRow(QLabel("Main Path | Date (YYYY-MM-DD) | Set Date:"), top_row)

        # Folder selection list + session log (right side)
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.folder_list.setDisabled(True)  # enabled after Set Date succeeds

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        folders_and_log = QWidget()
        folders_and_log_layout = QHBoxLayout(folders_and_log)
        folders_and_log_layout.setContentsMargins(0, 0, 0, 0)
        folders_and_log_layout.addWidget(self.folder_list, 3)  # make folder list the more flexible area
        folders_and_log_layout.addWidget(self.log_view, 2)

        # Size policies: folder list expands; log expands but less; plot area below has a minimum height
        self.folder_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_view.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        form.addRow(QLabel("Folders in selected date:"), folders_and_log)

        self.select_btn = QPushButton("Select folders")
        self.select_btn.setDisabled(True)
        self.select_btn.clicked.connect(self.on_select_folders)
        form.addRow(self.select_btn)

        # Processing controls
        self.lb_spin = QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 10.0)
        self.lb_spin.setSingleStep(0.1)
        self.lb_spin.setValue(1.0)

        self.ppm_start = QDoubleSpinBox()
        self.ppm_start.setRange(0.0, 300.0)
        self.ppm_start.setSingleStep(0.1)
        self.ppm_start.setValue(175.0)

        self.ppm_stop = QDoubleSpinBox()
        self.ppm_stop.setRange(0.0, 300.0)
        self.ppm_stop.setSingleStep(0.1)
        self.ppm_stop.setValue(160.0)

        form.addRow("Apodization (lb):", self.lb_spin)
        form.addRow("PPM Start:", self.ppm_start)
        form.addRow("PPM Stop:", self.ppm_stop)

        # Hidden mode: implicit (based on number of selected folders)

        # Transient range inputs (kept visible)
        self.transient_start = QLineEdit("1")
        self.transient_stop = QLineEdit("50")
        form.addRow("Transient Start:", self.transient_start)
        form.addRow("Transient Stop:", self.transient_stop)

        root.addLayout(form)

        # Run controls
        btns = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.on_run)
        self.reint_btn = QPushButton("Re-integrate")
        self.reint_btn.clicked.connect(self.on_reintegrate)
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.on_next)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.reint_btn)
        btns.addStretch(1)
        btns.addWidget(self.prev_btn)
        btns.addWidget(self.next_btn)
        root.addLayout(btns)

        # Phase controls
        phase_row = QHBoxLayout()
        self.decr_phase_btn = QPushButton("Phase -")
        self.incr_phase_btn = QPushButton("Phase +")
        self.phase_dial = QDial()
        self.phase_dial.setRange(1, 10)
        self.phase_dial.setValue(1)
        self.phase_dial.setNotchesVisible(True)
        self.phase_label = QLabel("1")
        self.phase_dial.valueChanged.connect(lambda v: self.phase_label.setText(str(v)))
        self.decr_phase_btn.clicked.connect(self.decrease_phase)
        self.incr_phase_btn.clicked.connect(self.increase_phase)
        phase_row.addWidget(self.decr_phase_btn)
        phase_row.addWidget(self.incr_phase_btn)
        phase_row.addWidget(QLabel("Phase Step:"))
        phase_row.addWidget(self.phase_dial)
        phase_row.addWidget(self.phase_label)
        root.addLayout(phase_row)

        # Plot section in its own container so it keeps a minimum height
        self.canvas = None
        self.toolbar = None
        self.plot_container = QWidget()
        self.plot_container.setMinimumHeight(420)  # <- prevents the spectrum from getting crushed
        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.plot_container)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        self.layout = root  # for reuse in update_canvas()

    # ---------------- Actions ----------------

    def log(self, text: str):
        """Append a line to the right-side session log."""
        self.log_view.append(text)

    def _resolved_date_path(self):
        base = self.path_edit.text().strip()
        date = self.date_edit.text().strip()
        return os.path.join(base, date)

    def on_set_date(self):
        date_path = self._resolved_date_path()
        if not os.path.isdir(date_path):
            self.folder_list.clear()
            self.folder_list.setDisabled(True)
            self.select_btn.setDisabled(True)
            self.log(f"[Set Date] Date folder not found:\n{date_path}")
            return

        # List immediate subfolders
        subs = [d for d in sorted(os.listdir(date_path)) if os.path.isdir(os.path.join(date_path, d))]
        self.folder_list.clear()
        for name in subs:
            QListWidgetItem(name, self.folder_list)

        self.folder_list.setDisabled(False)
        self.select_btn.setDisabled(False)
        self.log(f"[Set Date] Resolved date folder:\n{date_path}\nFound {len(subs)} subfolder(s). Select one or more and click 'Select folders'.")

    def on_select_folders(self):
        items = self.folder_list.selectedItems()
        self.selected_folders = [it.text() for it in items]
        date_path = self._resolved_date_path()
        if not self.selected_folders:
            self.log("[Select folders] No folders selected. Please choose at least one.")
            return
        msg = "\n".join(self.selected_folders)
        self.log(f"[Select folders] Date path:\n{date_path}\nSelected folders:\n{msg}")

    def _clear_plot(self):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas = None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
            self.toolbar = None

    def on_run(self):
        # Validate date path and selections
        date_path = self._resolved_date_path()
        if not os.path.isdir(date_path):
            self.log(f"[Run] Date folder not found:\n{date_path}")
            return
        if not self.selected_folders:
            self.log("[Run] Please select folder(s) (and click 'Select folders') before running.")
            return

        # Determine mode implicitly
        mode = "Multi-Transient" if len(self.selected_folders) == 1 else "Multi-Experiment"
        self.log(f"[Run] Mode: {mode}")

        # Load data
        self.data = []
        self.phase_offsets = []
        self.figs = []
        self.current = 0

        lb = float(self.lb_spin.value())
        ppm_start = float(self.ppm_start.value())
        ppm_stop = float(self.ppm_stop.value())

        try:
            if mode == "Multi-Experiment":
                # Deterministic order over the user-selected folders
                for folder_name in sorted(self.selected_folders):
                    fid_path = os.path.join(date_path, folder_name, "1", "fid.csv")
                    if not os.path.isfile(fid_path):
                        raise FileNotFoundError(f"Missing fid.csv: {fid_path}")
                    fid, time = read_fid_csv(fid_path)
                    self.data.append((fid, time, folder_name, 1))  # (fid, time, timestamp, transient_num)
                    self.phase_offsets.append(0.0)
                    fig = process_and_plot(fid, time, lb, 0.0, ppm_start, ppm_stop)
                    self.figs.append(fig)
                    self.log(f"[Run] Loaded {fid_path}")

            else:  # Multi-Transient
                folder_name = self.selected_folders[0]
                target_path = os.path.join(date_path, folder_name)

                # Parse transient range
                try:
                    t_start = int(self.transient_start.text())
                    t_stop = int(self.transient_stop.text())
                except ValueError:
                    raise ValueError("Transient Start/Stop must be integers.")
                if t_stop < t_start:
                    raise ValueError("Transient Stop must be >= Transient Start.")

                # STRICT numeric ordering of subfolders
                numeric_subs = sorted(
                    int(s) for s in os.listdir(target_path) if s.isdigit() and os.path.isdir(os.path.join(target_path, s))
                )
                if not numeric_subs:
                    raise FileNotFoundError(f"No numeric subfolders found in {target_path}")

                for n in numeric_subs:
                    if t_start <= n <= t_stop:
                        fid_path = os.path.join(target_path, str(n), "fid.csv")
                        if not os.path.isfile(fid_path):
                            # Skip missing
                            self.log(f"[Run] Skipping missing {fid_path}")
                            continue
                        fid, time = read_fid_csv(fid_path)
                        self.data.append((fid, time, folder_name, n))
                        self.phase_offsets.append(0.0)
                        fig = process_and_plot(fid, time, lb, 0.0, ppm_start, ppm_stop)
                        self.figs.append(fig)
                        self.log(f"[Run] Loaded {fid_path}")

            if not self.data:
                raise RuntimeError("No datasets loaded.")

            self.update_canvas()  # draws current index 0
            self.log("[Run] Plot updated.")

        except Exception as e:
            self.log(f"[Run Error] {str(e)}")
            self._clear_plot()
            self.status_label.setText("Ready")

    def on_reintegrate(self):
        if not self.data:
            self.log("[Re-integrate] Nothing to re-integrate. Load data first.")
            return
        lb = float(self.lb_spin.value())
        ppm_start = float(self.ppm_start.value())
        ppm_stop = float(self.ppm_stop.value())
        fid, time, _, _ = self.data[self.current]
        offset = float(self.phase_offsets[self.current])
        fig = process_and_plot(fid, time, lb, offset, ppm_start, ppm_stop)
        self.figs[self.current] = fig
        self._redraw_current()
        self.log("[Re-integrate] Updated current trace.")

    def _redraw_current(self):
        # Remove existing widgets in the plot area
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas = None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
            self.toolbar = None

        # Add canvas + toolbar for current figure
        self.canvas = FigureCanvas(self.figs[self.current])
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()

        # Update status label
        _, _, ts_name, trans_num = self.data[self.current]
        self.status_label.setText(f"Timestamp: {ts_name} | Transient: {trans_num}")

    def update_canvas(self):
        """Refresh the current index view after initial run or phase changes."""
        self._redraw_current()

    def on_prev(self):
        if not self.data:
            return
        self.current = (self.current - 1) % len(self.data)
        self._redraw_current()

    def on_next(self):
        if not self.data:
            return
        self.current = (self.current + 1) % len(self.data)
        self._redraw_current()

    def increase_phase(self):
        if not self.data:
            return
        step = float(self.phase_dial.value())
        self.phase_offsets[self.current] += step
        self.on_reintegrate()

    def decrease_phase(self):
        if not self.data:
            return
        step = float(self.phase_dial.value())
        self.phase_offsets[self.current] -= step
        self.on_reintegrate()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    sys.exit(app.exec_())
