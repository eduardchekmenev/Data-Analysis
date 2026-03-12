# ============================================================
# BLOCK 1 — Imports and Model Definition
# ============================================================

import sys
import os
import io
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QRadioButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

# ============================================================
# Michaelis-Menten Model
# ============================================================
def michaelis_menten(x, Vmax, Km):
    return (Vmax * x) / (Km + x)

# ============================================================
# BLOCK 2 — Core Fitting Function
# ============================================================

def fit_mm(x, y, ci_level=95, fix_vmax=None, fix_km=None, use_weighted=False):
    """
    Perform Michaelis–Menten fit and return parameters, CIs, and fit statistics.
    
    Parameters:
        x, y: Data arrays
        ci_level: Confidence interval level (default 95%)
        fix_vmax: If provided, fix Vmax to this value
        fix_km: If provided, fix Km to this value
        use_weighted: If True, use weighted least squares with weights = 1/y
                      (appropriate for constant relative error in enzyme kinetics)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 5:
        raise ValueError("Need at least 5 valid data points for fitting.")

    # Define flexible fitting model
    if fix_vmax is not None and fix_km is None:
        def model(x, Km):
            return michaelis_menten(x, fix_vmax, Km)
        p0 = [np.median(x)]
    elif fix_km is not None and fix_vmax is None:
        def model(x, Vmax):
            return michaelis_menten(x, Vmax, fix_km)
        p0 = [np.max(y)]
    else:
        model = michaelis_menten
        p0 = [np.max(y), np.median(x)]

    # Set up weighting for curve_fit
    # For weighted fitting: sigma = y means we minimize sum of ((y-yhat)/y)^2 = sum of (relative_error)^2
    # For unweighted fitting: don't pass sigma args at all (let curve_fit use defaults)
    if use_weighted:
        # Weight by y to account for constant relative error
        # Points with smaller y get higher weight (smaller sigma)
        sigma = y.copy()
        # Prevent division by zero for very small y values
        sigma[sigma < 1e-10] = 1e-10
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(0, np.inf), maxfev=10000,
                               sigma=sigma, absolute_sigma=False)
    else:
        # Unweighted: use default curve_fit behavior (no sigma arguments)
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(0, np.inf), maxfev=10000)
    yhat = model(x, *popt)
    resid = y - yhat

    # Goodness-of-fit metrics
    n, k = len(y), len(popt)
    sse = np.sum(resid ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst
    rmse = np.sqrt(sse / n)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)

    # Confidence intervals
    alpha = 1 - ci_level / 100
    dof = max(1, n - k)
    tval = stats.t.ppf(1 - alpha / 2, dof)
    se = np.sqrt(np.diag(pcov))
    ci_low = popt - tval * se
    ci_high = popt + tval * se

    # Calculate catalytic efficiency (Vmax/Km) and its uncertainty
    # Using error propagation: SE(Vmax/Km) = (Vmax/Km) * sqrt((SE_Vmax/Vmax)^2 + (SE_Km/Km)^2 - 2*cov/(Vmax*Km))
    if fix_vmax is None and fix_km is None:
        Vmax, Km = popt[0], popt[1]
        se_vmax, se_km = se[0], se[1]
        cov_vmax_km = pcov[0, 1]
        
        cat_eff = Vmax / Km
        # Error propagation for ratio, accounting for covariance
        rel_var = (se_vmax/Vmax)**2 + (se_km/Km)**2 - 2*cov_vmax_km/(Vmax*Km)
        # Ensure non-negative before sqrt (can be negative if high positive correlation)
        rel_var = max(0, rel_var)
        se_cat_eff = cat_eff * np.sqrt(rel_var)
        ci_cat_eff_low = cat_eff - tval * se_cat_eff
        ci_cat_eff_high = cat_eff + tval * se_cat_eff
    else:
        # If one parameter is fixed, catalytic efficiency calculation is simpler
        if fix_vmax is not None:
            Km = popt[0]
            cat_eff = fix_vmax / Km
            se_cat_eff = (fix_vmax / Km**2) * se[0]
        else:  # fix_km is not None
            Vmax = popt[0]
            cat_eff = Vmax / fix_km
            se_cat_eff = se[0] / fix_km
        ci_cat_eff_low = cat_eff - tval * se_cat_eff
        ci_cat_eff_high = cat_eff + tval * se_cat_eff

    return {
        "params": popt,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "r2": r2,
        "rmse": rmse,
        "sse": sse,
        "aic": aic,
        "bic": bic,
        "yhat": yhat,
        "resid": resid,
        "x": x,
        "y": y,
        "cat_eff": cat_eff,
        "se_cat_eff": se_cat_eff,
        "ci_cat_eff_low": ci_cat_eff_low,
        "ci_cat_eff_high": ci_cat_eff_high,
        "weighted": use_weighted
    }

# ============================================================
# BLOCK 3 — Matplotlib Canvas Class
# ============================================================

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def plot_fit(self, x, y, popt, max_x_display=None, max_y_display=None, max_extrap=None, 
                 cat_eff=None, weighted=False):
        """Plot data and fitted curve with optional axis limits and markers."""
        self.ax.clear()
        self.ax.scatter(x, y, label="Data", color="blue")

        vmax, km = popt
        if max_extrap is None:
            max_extrap = max(x) * 1.1
        xx = np.linspace(0, max_extrap, 400)
        yy = michaelis_menten(xx, *popt)
        self.ax.plot(xx, yy, color="orange", label="Fit")

        # Vmax and 0.5 Vmax lines
        self.ax.axhline(vmax, color="red", ls="--", lw=1.2)
        self.ax.text(max_extrap * 0.8, vmax, f"Vmax = {vmax:.3g}", color="red",
                    va="bottom", ha="right", fontsize=9)
        self.ax.axhline(vmax / 2, color="gray", ls=":", lw=1.0)
        # Km vertical line
        self.ax.axvline(km, color="green", ls="--", lw=1.2)
        self.ax.text(km, vmax * 0.05, f"Km = {km:.3g} mM", color="green",
                    va="bottom", ha="left", rotation=90, fontsize=9)

        # Axis limits if provided
        if max_x_display:
            self.ax.set_xlim(0, max_x_display)
        if max_y_display:
            self.ax.set_ylim(0, max_y_display)

        self.ax.set_xlabel("Concentration (mM)")
        self.ax.set_ylabel("v0 (normalized units/time)")
        self.ax.legend()
        
        # Title includes catalytic efficiency and weighting info
        title = "Michaelis–Menten Fit"
        if weighted:
            title += " (Weighted)"
        if cat_eff is not None:
            title += f"\nVmax/Km = {cat_eff:.4g}"
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.draw()


# ============================================================
# BLOCK 4 — GUI Class and Layout
# ============================================================
class PasteAwareTable(QTableWidget):
    """Custom table widget that handles Ctrl+V paste with row limit and log messages."""
    def __init__(self, rows=10, cols=2, parent=None, log_callback=None):
        super().__init__(rows, cols, parent)
        self.log_callback = log_callback

    def keyPressEvent(self, event):
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QApplication, QTableWidgetItem
        if event.matches(QKeySequence.Paste):
            clipboard = QApplication.clipboard()
            text = clipboard.text().strip()
            if not text:
                return

            rows = [r for r in text.splitlines() if r.strip()]
            max_rows = self.rowCount()
            cols = self.columnCount()

            for i, line in enumerate(rows[:max_rows]):
                # Split by commas, tabs, or spaces
                parts = [p.strip() for p in line.replace('\t', ' ').replace(',', ' ').split() if p.strip()]
                if not parts:
                    continue

                # If single-column paste (e.g., one value per line)
                if len(parts) == 1:
                    self.setItem(i, 0, QTableWidgetItem(parts[0]))
                else:
                    for j in range(min(len(parts), cols)):
                        self.setItem(i, j, QTableWidgetItem(parts[j]))

            # Warn if too many rows in clipboard
            if len(rows) > max_rows and self.log_callback:
                self.log_callback("⚠ Table size insufficient for clipboard. Extra rows ignored.")
        else:
            super().keyPressEvent(event)

class MMFitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Michaelis–Menten Fit GUI")
        self.setMinimumSize(1200, 750)

        self.central = QWidget()
        self.setCentralWidget(self.central)
        main_layout = QVBoxLayout(self.central)

        # --- Table + Plot ---
        top_layout = QHBoxLayout()
        self.table = PasteAwareTable(10, 2, log_callback=self.log)
        self.table.setHorizontalHeaderLabels(["Concentration (mM)", "v0 (normalized)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.canvas = MplCanvas()
        top_layout.addWidget(self.table, 3)
        top_layout.addWidget(self.canvas, 5)

        # --- Table row controls ---
        row_control_layout = QHBoxLayout()
        row_control_layout.addWidget(QLabel("Number of Rows:"))
        self.row_spin = QSpinBox()
        self.row_spin.setRange(5, 500)
        self.row_spin.setValue(10)
        row_control_layout.addWidget(self.row_spin)

        self.set_rows_btn = QPushButton("Set Rows")
        self.set_rows_btn.clicked.connect(self.set_table_rows)
        row_control_layout.addWidget(self.set_rows_btn)

        self.clear_btn = QPushButton("Clear Table")
        self.clear_btn.clicked.connect(self.clear_table)
        row_control_layout.addWidget(self.clear_btn)
        row_control_layout.addStretch(1)

        # --- Display options layout ---
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("Max X display (mM):"))
        self.max_x_display = QDoubleSpinBox()
        self.max_x_display.setRange(0, 1e6)
        self.max_x_display.setDecimals(2)
        self.max_x_display.setValue(0)
        display_layout.addWidget(self.max_x_display)

        display_layout.addWidget(QLabel("Max Y display:"))
        self.max_y_display = QDoubleSpinBox()
        self.max_y_display.setRange(0, 1e6)
        self.max_y_display.setDecimals(3)
        self.max_y_display.setValue(0)
        display_layout.addWidget(self.max_y_display)

        display_layout.addWidget(QLabel("Max extrapolation (mM):"))
        self.max_extrap = QDoubleSpinBox()
        self.max_extrap.setRange(0, 1e6)
        self.max_extrap.setDecimals(2)
        self.max_extrap.setValue(0)
        display_layout.addWidget(self.max_extrap)

        # --- Fitting options ---
        fitting_options_layout = QHBoxLayout()
        self.chk_weighted = QCheckBox("Weighted Least Squares (relative error)")
        self.chk_weighted.setToolTip(
            "Use weighted fitting where weights = 1/y.\n"
            "Appropriate when measurement error is proportional to signal magnitude\n"
            "(constant relative/percent error), which is common in enzyme kinetics."
        )
        fitting_options_layout.addWidget(self.chk_weighted)
        fitting_options_layout.addStretch(1)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(row_control_layout)
        main_layout.addLayout(display_layout)
        main_layout.addLayout(fitting_options_layout)

        # --- Output controls ---
        control_layout = QGridLayout()

        self.btn_fit = QPushButton("Fit && Plot")
        self.btn_save = QPushButton("Save Selected Outputs")
        self.btn_folder = QPushButton("Select Output Folder")
        self.folder_line = QLineEdit()
        self.folder_line.setPlaceholderText("Select or enter output folder...")

        control_layout.addWidget(self.btn_fit, 0, 0)
        control_layout.addWidget(self.btn_save, 0, 1)
        control_layout.addWidget(self.folder_line, 1, 0, 1, 2)
        control_layout.addWidget(self.btn_folder, 1, 2)

        # --- Checkboxes for file outputs ---
        self.chk_param = QCheckBox("Parameters + CIs + GOF (CSV)")
        self.chk_curve = QCheckBox("Smooth fitted curve (CSV)")
        self.chk_pointwise = QCheckBox("Pointwise fit & residuals (CSV)")
        self.chk_png = QCheckBox("Plot PNG")
        self.chk_pdf = QCheckBox("Plot PDF")
        self.chk_param.setChecked(True)
        self.chk_png.setChecked(True)

        row = 2
        for cb in [self.chk_param, self.chk_curve, self.chk_pointwise, self.chk_png, self.chk_pdf]:
            control_layout.addWidget(cb, row, 0)
            row += 1

        main_layout.addLayout(control_layout)

        # --- Log box ---
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(QLabel("Log Messages:"))
        main_layout.addWidget(self.log_box)

        # Connect buttons
        self.btn_fit.clicked.connect(self.run_fit)
        self.btn_save.clicked.connect(self.save_outputs)
        self.btn_folder.clicked.connect(self.select_folder)

        self.output_folder = ""
        self.last_fit = None

# ============================================================
# BLOCK 5 — Event Logic and File Saving
# ============================================================

    def log(self, msg):
        ts = datetime.now().strftime("[%H:%M:%S] ")
        self.log_box.appendPlainText(ts + msg)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.folder_line.setText(folder)
            self.log(f"Output folder selected: {folder}")

    def get_table_data(self):
        data = []
        for row in range(self.table.rowCount()):
            vals = []
            for col in range(2):
                item = self.table.item(row, col)
                if item and item.text().strip():
                    vals.append(item.text().strip())
            if len(vals) == 2:
                try:
                    data.append([float(vals[0]), float(vals[1])])
                except ValueError:
                    continue
        return np.array(data)

    def set_table_rows(self):
        """Reset table to the number of rows chosen in the spin box."""
        n = self.row_spin.value()
        self.table.setRowCount(n)
        self.log(f"✓ Table row count set to {n}.")

    def clear_table(self):
        """Clear all cells while keeping current row count."""
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem(""))
        self.log("✓ Table cleared.")

    def run_fit(self):
        try:
            data = self.get_table_data()
            if len(data) < 5:
                self.log("✖ Not enough valid data points.")
                return
            x, y = data[:, 0], data[:, 1]
            
            # Check if weighted fitting is requested
            use_weighted = self.chk_weighted.isChecked()
            
            res = fit_mm(x, y, use_weighted=use_weighted)
            self.last_fit = res
            
            # Collect display limits
            max_x_disp = self.max_x_display.value() or None
            max_y_disp = self.max_y_display.value() or None
            max_extrap = self.max_extrap.value() or None

            self.canvas.plot_fit(x, y, res["params"],
                                max_x_display=max_x_disp,
                                max_y_display=max_y_disp,
                                max_extrap=max_extrap,
                                cat_eff=res["cat_eff"],
                                weighted=use_weighted)
            
            # Log results including catalytic efficiency
            weight_str = " (weighted)" if use_weighted else ""
            self.log(f"✓ Fit complete{weight_str}:")
            self.log(f"    Vmax = {res['params'][0]:.4g} ± {res['se'][0]:.4g}")
            self.log(f"    Km   = {res['params'][1]:.4g} ± {res['se'][1]:.4g}")
            self.log(f"    Vmax/Km (catalytic efficiency) = {res['cat_eff']:.4g} ± {res['se_cat_eff']:.4g}")
            self.log(f"    R² = {res['r2']:.4f}")
            
        except Exception as e:
            self.log(f"✖ Fit failed: {e}")

    def save_outputs(self):
        if not self.last_fit:
            self.log("✖ No fit results to save.")
            return
        if not self.output_folder:
            self.log("✖ Please select an output folder first.")
            return

        base = f"mmfit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        folder = self.output_folder

        res = self.last_fit
        x, y, yhat, resid = res["x"], res["y"], res["yhat"], res["resid"]
        xx = np.linspace(0, max(x) * 1.1, 400)
        yy = michaelis_menten(xx, *res["params"])

        if self.chk_param.isChecked():
            # Parameters table now includes catalytic efficiency
            df_param = pd.DataFrame({
                "Parameter": ["Vmax", "Km (mM)", "Vmax/Km (catalytic efficiency)"],
                "Estimate": [res["params"][0], res["params"][1], res["cat_eff"]],
                "SE": [res["se"][0], res["se"][1], res["se_cat_eff"]],
                "CI low": [res["ci_low"][0], res["ci_low"][1], res["ci_cat_eff_low"]],
                "CI high": [res["ci_high"][0], res["ci_high"][1], res["ci_cat_eff_high"]]
            })
            df_gof = pd.DataFrame([{
                "R2": res["r2"], "RMSE": res["rmse"], "SSE": res["sse"],
                "AIC": res["aic"], "BIC": res["bic"], "Weighted": res["weighted"]
            }])
            df_param.to_csv(os.path.join(folder, base + "_parameters.csv"), index=False)
            df_gof.to_csv(os.path.join(folder, base + "_gof.csv"), index=False)
            self.log("✓ Saved parameters (incl. catalytic efficiency) and GOF CSV.")

        if self.chk_curve.isChecked():
            pd.DataFrame({"x": xx, "MM_fit": yy}).to_csv(
                os.path.join(folder, base + "_curve.csv"), index=False)
            self.log("✓ Saved smooth fitted curve CSV.")

        if self.chk_pointwise.isChecked():
            pd.DataFrame({"x": x, "y": y, "yhat": yhat, "residual": resid}).to_csv(
                os.path.join(folder, base + "_pointwise.csv"), index=False)
            self.log("✓ Saved pointwise fit and residuals CSV.")

        if self.chk_png.isChecked() or self.chk_pdf.isChecked():
            fig_path = os.path.join(folder, base + "_plot")
            self.canvas.fig.savefig(fig_path + ".png", dpi=300)
            if self.chk_pdf.isChecked():
                self.canvas.fig.savefig(fig_path + ".pdf")
            self.log("✓ Saved plot PNG/PDF.")

# ============================================================
# BLOCK 6 — Run Application
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MMFitGUI()
    win.show()
    sys.exit(app.exec())
