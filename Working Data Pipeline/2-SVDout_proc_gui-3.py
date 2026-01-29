# ===== Standard Library =====
import sys
import os
from pathlib import Path
from datetime import datetime

# ===== Third-Party =====
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")  # ensure Qt backend
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QTextEdit, QHBoxLayout, QMessageBox,
    QCheckBox, QLabel, QLineEdit
)

import re


# ----------------------------
# Utility functions
# ----------------------------
def extract_exptdate(filename: str):
    """Extract the 6-digit experiment date from filename after 'integrated_data_'."""
    m = re.search(r"integrated_data_(\d{6})", filename)
    return m.group(1) if m else None


def find_integral_columns(csv_path, metabolite_names):
    """
    Parse CSV header rows to find columns containing 'Integral' AND a matching metabolite name.
    
    CSV structure:
      - Row 0: 'Height'/'Integral' flags
      - Row 1: ppm values (ignored for matching)
      - Row 2: substrate names (e.g., 'lactate', 'pyruvate', 'bicarbonate', 'CO2')
      - Row 3+: numeric data
      - Col 0: Time
    
    Returns:
        dict mapping metabolite_name -> column_index (0-based, in the raw CSV)
    """
    raw = pd.read_csv(csv_path, header=None, nrows=3)
    
    row0 = raw.iloc[0]  # Height/Integral
    row2 = raw.iloc[2]  # Substrate names
    
    found_columns = {}
    
    for metabolite in metabolite_names:
        metabolite_lower = metabolite.strip().lower()
        
        for col_idx in range(1, len(row0)):  # Skip column 0 (Time)
            col_type = str(row0.iloc[col_idx]).strip().lower()
            col_substrate = str(row2.iloc[col_idx]).strip().lower()
            
            if col_type == 'integral' and col_substrate == metabolite_lower:
                found_columns[metabolite.strip()] = col_idx
                break
    
    return found_columns


def compute_metrics(csv_path,
                    metabolite_names,
                    xlim=None,
                    ylim=None,
                    export_png=False,
                    export_pdf=False,
                    export_csv=False,
                    output_dir: Path = None):
    """
    Loads a CSV and computes peak metrics for each specified metabolite.
    
    Args:
        csv_path: Path to the CSV file
        metabolite_names: List of metabolite names to analyze
        xlim, ylim: Plot limits
        export_png, export_pdf, export_csv: Export flags
        output_dir: Output directory for exports
    
    Returns:
        dict with metrics for each metabolite
    """
    # Find integral columns for each metabolite
    integral_columns = find_integral_columns(csv_path, metabolite_names)
    
    if not integral_columns:
        raise ValueError(f"No matching integral columns found for metabolites: {metabolite_names}")
    
    # Load the full data (skip first 3 header rows)
    raw = pd.read_csv(csv_path, header=None, skiprows=3)
    
    # Column 0 is Time
    time = pd.to_numeric(raw.iloc[:, 0], errors='coerce')
    
    # Build results dict
    results = {"file": Path(csv_path).name}
    
    # Store data for plotting
    plot_data = {}
    
    for metabolite, col_idx in integral_columns.items():
        # Extract and convert to numeric
        values = pd.to_numeric(raw.iloc[:, col_idx], errors='coerce')
        
        # Find peak
        idx_peak = values.idxmax()
        peak_value = float(values.loc[idx_peak])
        time_at_peak = float(time.loc[idx_peak])
        
        # Add to results with metabolite-specific column names
        results[f"peak_{metabolite}_integral"] = peak_value
        results[f"time_at_peak_{metabolite}"] = time_at_peak
        results[f"idx_peak_{metabolite}"] = int(idx_peak)
        
        # Store for plotting
        plot_data[metabolite] = (time, values)
    
    # Report any metabolites that weren't found
    missing = set(metabolite_names) - set(integral_columns.keys())
    if missing:
        results["missing_metabolites"] = ", ".join(missing)
    
    # --- Plotting ---
    n_metabolites = len(integral_columns)
    if n_metabolites > 0:
        fig, axes = plt.subplots(1, n_metabolites, figsize=(5 * n_metabolites, 4), squeeze=False)
        axes = axes.flatten()
        
        for i, (metabolite, (t, vals)) in enumerate(plot_data.items()):
            ax = axes[i]
            ax.plot(t, vals, label=f"{metabolite} integral")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{metabolite.capitalize()} Integral")
            if xlim:
                ax.set_xlim(0, xlim)
            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.set_title(f"{metabolite.capitalize()}")
            ax.legend()
        
        fig.suptitle(Path(csv_path).name, fontsize=10)
        plt.tight_layout()
        
        # --- Exports ---
        csv_path = Path(csv_path)
        outdir = Path(output_dir) if output_dir is not None else csv_path.parent
        name = csv_path.stem
        
        # Pull from first "(" inclusive; else fallback to "_name"
        if "(" in name:
            name_part = name[name.index("("):]
        else:
            name_part = "_" + name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        
        # Try to extract the experiment date from filename
        exptdate = extract_exptdate(os.path.basename(csv_path)) or ""
        date_prefix = f"{exptdate}_" if exptdate else ""
        outstem = f"{date_prefix}{name_part}_proc_at{timestamp}"
        
        if export_csv:
            out_csv = outdir / f"{outstem}.csv"
            pd.DataFrame([results]).to_csv(out_csv, index=False)
        
        if export_png:
            out_png = outdir / f"{outstem}.png"
            fig.savefig(out_png, dpi=300)
        
        if export_pdf:
            out_pdf = outdir / f"{outstem}.pdf"
            fig.savefig(out_pdf)
        
        plt.close(fig)
    
    return results


# ----------------------------
# GUI
# ----------------------------
class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metabolite Peak Analysis GUI")
        self.resize(1000, 750)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # --- Top row: input/output folder buttons ---
        top = QHBoxLayout()
        self.folder_btn = QPushButton("Select Input Folder")
        self.folder_btn.clicked.connect(self.load_folder)
        top.addWidget(self.folder_btn)
        
        self.out_btn = QPushButton("Select Output Folder")
        self.out_btn.clicked.connect(self.select_output_folder)
        top.addWidget(self.out_btn)
        
        self.layout.addLayout(top)
        
        # --- Metabolite input ---
        metab_layout = QHBoxLayout()
        metab_layout.addWidget(QLabel("Metabolites (comma-separated):"))
        self.metabolite_input = QLineEdit("pyruvate, CO2, bicarbonate")
        self.metabolite_input.setPlaceholderText("e.g., lactate, pyruvate, bicarbonate, CO2")
        metab_layout.addWidget(self.metabolite_input)
        self.layout.addLayout(metab_layout)
        
        # --- Export options ---
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Exports:"))
        self.cb_png = QCheckBox("Plot PNG")
        self.cb_pdf = QCheckBox("Plot PDF")
        self.cb_csv_each = QCheckBox("Per-file CSV")
        self.cb_csv_summary = QCheckBox("Summary CSV")
        
        # Defaults
        self.cb_png.setChecked(True)
        self.cb_pdf.setChecked(False)
        self.cb_csv_each.setChecked(False)
        self.cb_csv_summary.setChecked(True)
        
        for cb in (self.cb_png, self.cb_pdf, self.cb_csv_each, self.cb_csv_summary):
            opts.addWidget(cb)
        
        self.layout.addLayout(opts)
        
        # --- Plot limits (optional) ---
        limits_layout = QHBoxLayout()
        limits_layout.addWidget(QLabel("X limit:"))
        self.xlim_input = QLineEdit("400")
        self.xlim_input.setFixedWidth(60)
        limits_layout.addWidget(self.xlim_input)
        
        limits_layout.addWidget(QLabel("Y limit:"))
        self.ylim_input = QLineEdit("")
        self.ylim_input.setPlaceholderText("auto")
        self.ylim_input.setFixedWidth(60)
        limits_layout.addWidget(self.ylim_input)
        
        limits_layout.addStretch()
        self.layout.addLayout(limits_layout)
        
        # --- File list ---
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(self.file_list.MultiSelection)
        self.layout.addWidget(self.file_list)
        
        # --- Run button ---
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.layout.addWidget(self.run_btn)
        
        # --- Output log ---
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)
        
        # Remember selected output folder
        self.output_folder = None
    
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if not folder:
            return
        self.file_list.clear()
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                self.file_list.addItem(str(Path(folder) / f))
        self.output.append(f"Loaded input folder: {folder}")
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = Path(folder)
            self.output.append(f"Output folder set: {self.output_folder}")
    
    def run_analysis(self):
        selected = [item.text() for item in self.file_list.selectedItems()]
        if not selected:
            QMessageBox.warning(self, "No file selected", "Please select at least one CSV.")
            return
        
        # Parse metabolite names
        metabolite_text = self.metabolite_input.text().strip()
        if not metabolite_text:
            QMessageBox.warning(self, "No metabolites", "Please enter at least one metabolite name.")
            return
        
        metabolite_names = [m.strip() for m in metabolite_text.split(",") if m.strip()]
        if not metabolite_names:
            QMessageBox.warning(self, "No metabolites", "Please enter valid metabolite names.")
            return
        
        # Export options
        export_png = self.cb_png.isChecked()
        export_pdf = self.cb_pdf.isChecked()
        export_csv_each = self.cb_csv_each.isChecked()
        export_csv_summary = self.cb_csv_summary.isChecked()
        
        # If any export is enabled, require an output folder
        if (export_png or export_pdf or export_csv_each or export_csv_summary) and not self.output_folder:
            QMessageBox.warning(self, "No output folder", "Please select an output folder for exports.")
            return
        
        # Parse plot limits
        try:
            xlim = float(self.xlim_input.text()) if self.xlim_input.text().strip() else None
        except ValueError:
            xlim = None
        
        try:
            ylim = float(self.ylim_input.text()) if self.ylim_input.text().strip() else None
        except ValueError:
            ylim = None
        
        self.output.append(f"\nRunning analysis for metabolites: {metabolite_names}\n")
        rows = []
        
        for file in selected:
            try:
                res = compute_metrics(
                    file,
                    metabolite_names=metabolite_names,
                    xlim=xlim,
                    ylim=ylim,
                    export_png=export_png,
                    export_pdf=export_pdf,
                    export_csv=export_csv_each,
                    output_dir=self.output_folder
                )
                rows.append(res)
                
                # Print results for this file
                self.output.append(f"=== {res['file']} ===")
                for k, v in res.items():
                    if k != 'file':
                        if isinstance(v, float):
                            self.output.append(f"  {k}: {v:.6g}")
                        else:
                            self.output.append(f"  {k}: {v}")
                self.output.append("----")
            
            except Exception as e:
                self.output.append(f"ERROR: {Path(file).name}: {e}")
                import traceback
                self.output.append(traceback.format_exc())
        
        # Summary CSV
        if export_csv_summary and rows:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            
            # Extract date from first selected file
            first_file = selected[0] if selected else ""
            exptdate = extract_exptdate(os.path.basename(first_file)) or ""
            date_part = f"{exptdate}_" if exptdate else ""
            
            # Build summary name
            summary_path = self.output_folder / f"summary_{date_part}proc_at{timestamp}.csv"
            
            # Create DataFrame and order columns: file first, then metabolite metrics in order
            df = pd.DataFrame(rows)
            
            # Reorder columns: file first, then grouped by metabolite
            ordered_cols = ['file']
            for metab in metabolite_names:
                for suffix in ['_integral', '']:
                    col_integral = f"peak_{metab}_integral"
                    col_time = f"time_at_peak_{metab}"
                    col_idx = f"idx_peak_{metab}"
                    for col in [col_integral, col_time, col_idx]:
                        if col in df.columns and col not in ordered_cols:
                            ordered_cols.append(col)
            
            # Add any remaining columns (like missing_metabolites)
            for col in df.columns:
                if col not in ordered_cols:
                    ordered_cols.append(col)
            
            df = df[ordered_cols]
            df.to_csv(summary_path, index=False)
            self.output.append(f"\nSummary saved: {summary_path}")
        
        self.output.append("\nDone.\n")


# ----------------------------
# Entrypoint
# ----------------------------
def main():
    app = QApplication(sys.argv)
    gui = AnalysisGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
