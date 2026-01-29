#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def ppm_conversion(freqs):
    """
    Convert frequency (Hz) to ppm using user-specified mapping.
    ppm = -((freqs - (2500 - 639.08016399999997)) / 15.507665)
    """
    return -((freqs - (2500 - 639.08016399999997)) / 15.507665)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax_fid = self.fig.add_subplot(2, 1, 1)
        self.ax_fft = self.fig.add_subplot(2, 1, 2)
        super().__init__(self.fig)


class FIDFFTViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple FID/FFT Viewer (FID real; FFT real in ppm)")
        self.resize(1100, 750)

        # Data containers
        self.time_s = None
        self.fid_real = None
        self.fid_imag = None
        self.cumulative_phase_deg = 0.0  # zero-order phase accumulator
        self.current_basename = None
        self.input_folder = ""
        self.output_folder = ""

        # Central widget & layout
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # ---------- Left control panel ----------
        left = QtWidgets.QFrame()
        left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        left.setMinimumWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left)

        # Input folder
        self.btn_input = QtWidgets.QPushButton("Select Input Folder…")
        self.btn_input.clicked.connect(self.select_input_folder)
        left_layout.addWidget(self.btn_input)

        # File list
        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemSelectionChanged.connect(self.load_selected_file)
        left_layout.addWidget(self.file_list, 1)

        # FFT options group (ppm only)
        grp_fft = QtWidgets.QGroupBox("FFT Options (ppm only; ranges apply to FFT plot)")
        g_fft = QtWidgets.QGridLayout(grp_fft)

        self.le_xmin = QtWidgets.QLineEdit()
        self.le_xmax = QtWidgets.QLineEdit()
        self.le_ymin = QtWidgets.QLineEdit()
        self.le_ymax = QtWidgets.QLineEdit()

        # Placeholders to indicate auto
        for le, ph in [(self.le_xmin, "auto"), (self.le_xmax, "auto"),
                       (self.le_ymin, "auto"), (self.le_ymax, "auto")]:
            le.setPlaceholderText(ph)
            le.setMaximumWidth(120)
            le.editingFinished.connect(self.refresh_plots)

        g_fft.addWidget(QtWidgets.QLabel("X min (ppm)"), 0, 0)
        g_fft.addWidget(self.le_xmin, 0, 1)
        g_fft.addWidget(QtWidgets.QLabel("X max (ppm)"), 0, 2)
        g_fft.addWidget(self.le_xmax, 0, 3)
        g_fft.addWidget(QtWidgets.QLabel("Y min"), 1, 0)
        g_fft.addWidget(self.le_ymin, 1, 1)
        g_fft.addWidget(QtWidgets.QLabel("Y max"), 1, 2)
        g_fft.addWidget(self.le_ymax, 1, 3)

        # Apodization (lb in 1/s)
        self.lb_spin = QtWidgets.QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 1e6)
        self.lb_spin.setDecimals(6)
        self.lb_spin.setValue(0.0)
        self.lb_spin.setSingleStep(0.1)
        self.lb_spin.valueChanged.connect(self.refresh_plots)
        g_fft.addWidget(QtWidgets.QLabel("Apodization lb (1/s)"), 2, 0)
        g_fft.addWidget(self.lb_spin, 2, 1)

        left_layout.addWidget(grp_fft)

        # Phase control group
        grp_phase = QtWidgets.QGroupBox("Zero-Order Phase")
        g_phase = QtWidgets.QGridLayout(grp_phase)

        self.phase_dial = QtWidgets.QDial()
        self.phase_dial.setRange(1, 10)  # step size 1–10 degrees
        self.phase_dial.setNotchesVisible(True)
        self.phase_dial.setValue(1)
        self.phase_dial.setToolTip("Phase step size (degrees) for + / – buttons")

        self.btn_phase_minus = QtWidgets.QPushButton("–")
        self.btn_phase_plus = QtWidgets.QPushButton("+")
        self.btn_phase_minus.setFixedWidth(50)
        self.btn_phase_plus.setFixedWidth(50)
        self.btn_phase_minus.clicked.connect(lambda: self.apply_phase(sign=-1))
        self.btn_phase_plus.clicked.connect(lambda: self.apply_phase(sign=+1))

        self.lbl_phase = QtWidgets.QLabel("Current phase: 0.0°")
        self.lbl_phase.setAlignment(Qt.AlignCenter)

        g_phase.addWidget(QtWidgets.QLabel("Step (°)"), 0, 0)
        g_phase.addWidget(self.phase_dial, 0, 1, 3, 1)
        g_phase.addWidget(self.btn_phase_minus, 0, 2)
        g_phase.addWidget(self.btn_phase_plus, 1, 2)
        g_phase.addWidget(self.lbl_phase, 2, 2)

        left_layout.addWidget(grp_phase)

        # Output folder + save buttons
        self.btn_output = QtWidgets.QPushButton("Select Output Folder…")
        self.btn_output.clicked.connect(self.select_output_folder)
        left_layout.addWidget(self.btn_output)

        save_row = QtWidgets.QHBoxLayout()
        self.btn_save_fid = QtWidgets.QPushButton("Save FID (PNG+PDF)")
        self.btn_save_fft = QtWidgets.QPushButton("Save FFT (PNG+PDF)")
        self.btn_save_fid.clicked.connect(self.save_fid_plot)
        self.btn_save_fft.clicked.connect(self.save_fft_plot)
        save_row.addWidget(self.btn_save_fid)
        save_row.addWidget(self.btn_save_fft)
        left_layout.addLayout(save_row)

        left_layout.addStretch(1)

        # ---------- Right: log + plots ----------
        right = QtWidgets.QFrame()
        right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        right_layout = QtWidgets.QVBoxLayout(right)

        # Log/status box above plots
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        self.log.setStyleSheet("QTextEdit { background: #f6f6f7; }")
        right_layout.addWidget(self.log)

        # Canvas
        self.canvas = MplCanvas(self, width=7.5, height=6.0, dpi=100)
        right_layout.addWidget(self.canvas, 1)

        # Add to main layout
        main_layout.addWidget(left)
        main_layout.addWidget(right, 1)

        # Initial plot labels
        self._init_plot_labels()

    # -------------------- UI helpers --------------------

    def _init_plot_labels(self):
        self.canvas.ax_fid.clear()
        self.canvas.ax_fft.clear()

        self.canvas.ax_fid.set_title("FID (Real) vs Time")
        self.canvas.ax_fid.set_xlabel("Time (s)")
        self.canvas.ax_fid.set_ylabel("Amplitude (arb.)")

        self.canvas.ax_fft.set_title("FFT (Real) vs Chemical Shift")
        self.canvas.ax_fft.set_xlabel("Chemical Shift (ppm)")
        self.canvas.ax_fft.set_ylabel("Amplitude (arb.)")

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

    def log_msg(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # -------------------- Folder / file selection --------------------

    def select_input_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_folder or os.getcwd())
        if not folder:
            return
        self.input_folder = folder
        self.log_msg(f"<b>Input folder:</b> {folder}")
        self.populate_file_list()

    def populate_file_list(self):
        self.file_list.clear()
        if not self.input_folder:
            return
        try:
            files = [f for f in os.listdir(self.input_folder)
                     if os.path.isfile(os.path.join(self.input_folder, f))
                     and f.lower().endswith(".csv")
                     and "fid" in f.lower()]
            files.sort()
            for f in files:
                self.file_list.addItem(f)
            if not files:
                self.log_msg("No CSV files containing 'fid' found in the selected folder.")
        except Exception as e:
            self.log_msg(f"<span style='color:#b00;'>Error listing files: {e}</span>")
            self.log_msg(traceback.format_exc())

    def load_selected_file(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        filename = items[0].text()
        path = os.path.join(self.input_folder, filename)
        self.current_basename = os.path.splitext(filename)[0]
        self.log_msg(f"<b>Selected file:</b> {filename}")

        try:
            # Expect headers; time(s), real, imag
            df = pd.read_csv(path)

            # ---- Robust column mapping (supports e.g. 'denoised_real') ----
            orig_cols = list(df.columns)
            lower_to_orig = {c.strip().lower(): c for c in orig_cols}

            def pick_time_col():
                for key in ["time", "time_s", "t", "seconds", "sec", "time (s)", "time[s]"]:
                    if key in lower_to_orig:
                        return lower_to_orig[key]
                for lc, orig in lower_to_orig.items():
                    if ("time" in lc or "sec" in lc) and "index" not in lc:
                        return orig
                return None

            def pick_real_col():
                for key in ["real", "re", "fid_real", "fidre", "denoised_real"]:
                    if key in lower_to_orig:
                        return lower_to_orig[key]
                for lc, orig in lower_to_orig.items():
                    if "real" in lc and "imag" not in lc and "image" not in lc:
                        return orig
                return None

            def pick_imag_col():
                for key in ["imag", "im", "fid_imag", "fidim", "denoised_imag"]:
                    if key in lower_to_orig:
                        return lower_to_orig[key]
                for lc, orig in lower_to_orig.items():
                    if "imag" in lc:
                        return orig
                for lc, orig in lower_to_orig.items():
                    if lc.endswith("_im"):
                        return orig
                return None

            col_t = pick_time_col()
            col_r = pick_real_col()
            col_i = pick_imag_col()

            if col_t is None or col_r is None:
                raise ValueError(
                    "Could not find required columns for time and real. "
                    f"Available columns: {orig_cols}"
                )

            if col_i is None:
                self.log_msg("No imaginary column found; assuming imag = 0.")
                df["__imag__"] = 0.0
                col_i = "__imag__"
            else:
                self.log_msg(f"Mapped columns → time: '{col_t}', real: '{col_r}', imag: '{col_i}'")

            t = df[col_t].to_numpy(dtype=float)
            r = df[col_r].to_numpy(dtype=float)
            i = df[col_i].to_numpy(dtype=float)

            if len(t) < 2:
                raise ValueError("Not enough points in the file.")

            # Store data
            self.time_s = t
            self.fid_real = r
            self.fid_imag = i

            # Reset phase accumulator when a new file is loaded
            self.cumulative_phase_deg = 0.0
            self.lbl_phase.setText(f"Current phase: {self.cumulative_phase_deg:.1f}°")

            # Plot
            self.refresh_plots()
        except Exception as e:
            self.log_msg(f"<span style='color:#b00;'>Error loading file: {e}</span>")
            self.log_msg(traceback.format_exc())

    def select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder or os.getcwd())
        if not folder:
            return
        self.output_folder = folder
        self.log_msg(f"<b>Output folder:</b> {folder}")

    # -------------------- Processing & plotting --------------------

    def get_complex_fid_processed(self):
        """Return processed complex FID after apodization and phase."""
        if self.time_s is None or self.fid_real is None:
            return None, None
        t = self.time_s
        # Complex FID from real + imag
        fid = self.fid_real.astype(np.complex128) + 1j * self.fid_imag.astype(np.complex128)

        # Apodization: exp(-lb * t)
        lb = float(self.lb_spin.value())
        if lb != 0.0:
            fid = fid * np.exp(-lb * t)

        # Zero-order phase (degrees → radians)
        if self.cumulative_phase_deg != 0.0:
            phi = np.deg2rad(self.cumulative_phase_deg)
            fid = fid * np.exp(1j * phi)

        return t, fid

    def compute_fft_real(self, t, fid):
        """Compute frequency axis (Hz) and REAL part of FFT(fid) with fftshift."""
        dt = float(t[1] - t[0])  # assume uniform sampling
        n = len(t)
        spec = np.fft.fftshift(np.fft.fft(fid))
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
        real_spec = np.real(spec)  # plot real part only
        return freqs, real_spec

    def refresh_plots(self):
        # Clear axes
        self.canvas.ax_fid.clear()
        self.canvas.ax_fft.clear()

        # Titles and labels
        self.canvas.ax_fid.set_title("FID (Real) vs Time")
        self.canvas.ax_fid.set_xlabel("Time (s)")
        self.canvas.ax_fid.set_ylabel("Amplitude (arb.)")

        self.canvas.ax_fft.set_title("FFT (Real) vs Chemical Shift")
        self.canvas.ax_fft.set_xlabel("Chemical Shift (ppm)")
        self.canvas.ax_fft.set_ylabel("Amplitude (arb.)")

        try:
            if self.time_s is None or self.fid_real is None:
                self.canvas.draw_idle()
                return

            # FID (real only) – always full data
            self.canvas.ax_fid.plot(self.time_s, self.fid_real, lw=1.0)

            # FFT (real of spectrum) with apodization + phase
            t, fid_c = self.get_complex_fid_processed()
            freqs_hz, real_spec = self.compute_fft_real(t, fid_c)

            # Convert x-axis (Hz) to ppm
            x_vals_ppm = ppm_conversion(freqs_hz)

            self.canvas.ax_fft.plot(x_vals_ppm, real_spec, lw=1.0)

            # Apply user-specified FFT axis ranges (in ppm)
            def parse_float(le: QtWidgets.QLineEdit):
                txt = le.text().strip()
                return None if txt == "" else float(txt)

            xmin = parse_float(self.le_xmin)
            xmax = parse_float(self.le_xmax)
            ymin = parse_float(self.le_ymin)
            ymax = parse_float(self.le_ymax)

            if xmin is not None or xmax is not None:
                self.canvas.ax_fft.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self.canvas.ax_fft.set_ylim(bottom=ymin, top=ymax)

            self.canvas.ax_fft.invert_xaxis()

            self.canvas.fig.tight_layout()
            self.canvas.draw_idle()

        except Exception as e:
            self.log_msg(f"<span style='color:#b00;'>Plot error: {e}</span>")
            self.log_msg(traceback.format_exc())

    # -------------------- Phase controls --------------------

    def apply_phase(self, sign=+1):
        step = float(self.phase_dial.value())
        self.cumulative_phase_deg += sign * step
        self.lbl_phase.setText(f"Current phase: {self.cumulative_phase_deg:.1f}°")
        self.log_msg(f"Applied {'+' if sign>0 else '–'}{step:.1f}° → cumulative {self.cumulative_phase_deg:.1f}°")
        self.refresh_plots()

    # -------------------- Saving --------------------

    def ensure_output_folder(self):
        if not self.output_folder:
            QtWidgets.QMessageBox.warning(self, "No Output Folder", "Please select an output folder first.")
            return False
        if not os.path.isdir(self.output_folder):
            QtWidgets.QMessageBox.warning(self, "Invalid Output Folder", "The selected output folder does not exist.")
            return False
        return True

    def save_fid_plot(self):
        if not self.ensure_output_folder():
            return
        if self.time_s is None or self.fid_real is None or self.current_basename is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Load a file before saving.")
            return
        try:
            # Create a dedicated figure for FID
            fig = Figure(figsize=(7, 4), dpi=150, tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.time_s, self.fid_real, lw=1.0)
            ax.set_title("FID (Real) vs Time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (arb.)")

            png_path = os.path.join(self.output_folder, f"fid_{self.current_basename}.png")
            pdf_path = os.path.join(self.output_folder, f"fid_{self.current_basename}.pdf")
            fig.savefig(png_path)
            fig.savefig(pdf_path)
            self.log_msg(f"Saved FID: {png_path}")
            self.log_msg(f"Saved FID: {pdf_path}")
        except Exception as e:
            self.log_msg(f"<span style='color:#b00;'>Save FID error: {e}</span>")
            self.log_msg(traceback.format_exc())

    def save_fft_plot(self):
        if not self.ensure_output_folder():
            return
        if self.time_s is None or self.fid_real is None or self.current_basename is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Load a file before saving.")
            return
        try:
            # Prepare processed FFT data
            t, fid_c = self.get_complex_fid_processed()
            freqs_hz, real_spec = self.compute_fft_real(t, fid_c)

            # Convert to ppm
            x_vals_ppm = ppm_conversion(freqs_hz)

            # Create a dedicated figure for FFT in ppm
            fig = Figure(figsize=(7, 4), dpi=150, tight_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x_vals_ppm, real_spec, lw=1.0)
            ax.set_title("FFT (Real) vs Chemical Shift")
            ax.set_xlabel("Chemical Shift (ppm)")
            ax.set_ylabel("Amplitude (arb.)")

            # Apply axis constraints from UI (ppm)
            def parse_float(le: QtWidgets.QLineEdit):
                txt = le.text().strip()
                return None if txt == "" else float(txt)

            xmin = parse_float(self.le_xmin)
            xmax = parse_float(self.le_xmax)
            ymin = parse_float(self.le_ymin)
            ymax = parse_float(self.le_ymax)
            if xmin is not None or xmax is not None:
                ax.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                ax.set_ylim(bottom=ymin, top=ymax)

            ax.invert_xaxis()

            png_path = os.path.join(self.output_folder, f"fft_{self.current_basename}.png")
            pdf_path = os.path.join(self.output_folder, f"fft_{self.current_basename}.pdf")
            fig.savefig(png_path)
            fig.savefig(pdf_path)
            self.log_msg(f"Saved FFT: {png_path}")
            self.log_msg(f"Saved FFT: {pdf_path}")
        except Exception as e:
            self.log_msg(f"<span style='color:#b00;'>Save FFT error: {e}</span>")
            self.log_msg(traceback.format_exc())


def main():
    # High-DPI / scaling hints for better visuals
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    win = FIDFFTViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
