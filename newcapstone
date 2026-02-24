import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TrackManAnalysis:
    def __init__(self, root):
        self.root = root
        self.root.title("Trackman Pitching Analysis Development Program")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        # ===== HEADER =====
        header = tk.Frame(self.root, bg="#1f2c3c", pady=20)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="Trackman Pitching Analysis Dashboard",
            fg="white",
            bg="#1f2c3c",
            font=("Helvetica", 22, "bold")
        ).pack()

        # ===== CONTROL PANEL =====
        control_panel = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        control_panel.pack(fill=tk.X)

        tk.Button(
            control_panel,
            text="Import Trackman CSV",
            command=self.import_csv,
            font=("Helvetica", 12),
            bg="#2980b9",
            fg="white",
            padx=20,
            pady=8
        ).pack()

        # ===== SCROLLABLE AREA =====
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    # =============================
    # CSV IMPORT
    # =============================
    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.plot_dashboard(df)
            except Exception as error:
                messagebox.showerror("Error", f"Could not read CSV: {error}")

    # =============================
    # MAIN DASHBOARD
    # =============================
    def plot_dashboard(self, df):

        required_cols = ['pitch_name', 'release_speed']
        if not all(col in df.columns for col in required_cols):
            messagebox.showwarning(
                "Warning",
                "CSV must contain 'pitch_name' and 'release_speed'."
            )
            return

        df = df.dropna(subset=required_cols)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # =============================
        # CHART SECTION
        # =============================
        usage = df['pitch_name'].value_counts()
        avg_speeds = df.groupby('pitch_name')['release_speed'].mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plt.subplots_adjust(wspace=0.5)

        ax1.pie(usage, labels=usage.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Pitch Usage Mix (%)")

        bars = ax2.bar(avg_speeds.index, avg_speeds.values)
        ax2.set_title("Average Release Speed (MPH)")
        ax2.set_ylabel("MPH")

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}', ha='center', va='bottom')

        plt.setp(ax2.get_xticklabels(), rotation=30)

        chart_canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(pady=25)

        # =============================
        # DASHBOARD METRICS
        # =============================
        metrics_frame = tk.Frame(self.scrollable_frame, bg="white")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        def section_title(title):
            tk.Label(
                metrics_frame,
                text=title,
                font=("Helvetica", 18, "bold"),
                bg="white",
                fg="#1f2c3c",
                pady=15
            ).pack(anchor="w")

        def metric_card(parent, title, value, row, column, color="#2c3e50"):
            card = tk.Frame(parent, bg="#ecf0f1", bd=2, relief="ridge")
            card.grid(row=row, column=column, padx=25, pady=20, sticky="nsew")

            tk.Label(card, text=title,
                     font=("Helvetica", 11, "bold"),
                     bg="#ecf0f1",
                     fg=color).pack(padx=25, pady=(15, 5))

            tk.Label(card, text=value,
                     font=("Helvetica", 16),
                     bg="#ecf0f1").pack(padx=25, pady=(0, 15))

        # =============================
        # PERFORMANCE
        # =============================
        section_title("Performance Metrics")
        perf_frame = tk.Frame(metrics_frame, bg="white")
        perf_frame.pack(anchor="w")

        avg_vel = df['release_speed'].mean()
        vel_std = df['release_speed'].std()

        metric_card(perf_frame, "Avg Velocity", f"{avg_vel:.2f} MPH", 0, 0)
        metric_card(perf_frame, "Velocity Std Dev", f"{vel_std:.2f}", 0, 1)

        if 'release_spin_rate' in df.columns:
            metric_card(perf_frame, "Avg Spin Rate",
                        f"{df['release_spin_rate'].mean():.0f} RPM", 0, 2)

        # =============================
        # OPTIMAL METRICS COMPARISON
        # =============================
        section_title("Optimal Pitching Metrics Comparison (By Pitch Type)")

        optimal_container = tk.Frame(metrics_frame, bg="white")
        optimal_container.pack(fill=tk.BOTH, expand=True, pady=15)

        optimal_metrics_by_pitch = {
            "Fastball": {"Velocity": 95, "Spin Rate": 2300, "Horizontal Break": 2, "Vertical Break": 10},
            "4-Seam Fastball": {"Velocity": 96, "Spin Rate": 2350, "Horizontal Break": 1.5, "Vertical Break": 11},
            "Cutter": {"Velocity": 91, "Spin Rate": 2200, "Horizontal Break": 4, "Vertical Break": 8},
            "Sinker": {"Velocity": 92, "Spin Rate": 2250, "Horizontal Break": 3, "Vertical Break": 9},
            "Slider": {"Velocity": 87, "Spin Rate": 2500, "Horizontal Break": 5, "Vertical Break": 6},
            "Curveball": {"Velocity": 78, "Spin Rate": 2600, "Horizontal Break": 3, "Vertical Break": 8},
            "Changeup": {"Velocity": 83, "Spin Rate": 2100, "Horizontal Break": 2, "Vertical Break": 5},
            "Sweeper": {"Velocity": 85, "Spin Rate": 2450, "Horizontal Break": 6, "Vertical Break": 5}
        }

        recommendations = []
        row_index = 0

        for pitch in df['pitch_name'].unique():
            df_pitch = df[df['pitch_name'] == pitch]
            opt = optimal_metrics_by_pitch.get(pitch)

            pitch_box = tk.Frame(optimal_container, bg="#f1f2f6", bd=2, relief="solid")
            pitch_box.grid(row=row_index, column=0, sticky="ew", pady=15, padx=10)

            tk.Label(pitch_box,
                     text=pitch,
                     font=("Helvetica", 15, "bold"),
                     bg="#dcdde1").pack(fill=tk.X)

            content = tk.Frame(pitch_box, bg="#f1f2f6")
            content.pack(padx=25, pady=15)

            if opt:
                avg_vel_pitch = df_pitch['release_speed'].mean()
                tk.Label(content,
                         text=f"Velocity: {avg_vel_pitch:.1f} MPH | Optimal: {opt['Velocity']} MPH",
                         bg="#f1f2f6").pack(anchor="w", pady=4)

                if avg_vel_pitch < opt['Velocity']:
                    recommendations.append(f"- {pitch}: Improve lower-body power to increase velocity.")

                if 'release_spin_rate' in df.columns:
                    avg_spin = df_pitch['release_spin_rate'].mean()
                    tk.Label(content,
                             text=f"Spin Rate: {avg_spin:.0f} RPM | Optimal: {opt['Spin Rate']} RPM",
                             bg="#f1f2f6").pack(anchor="w", pady=4)

                    if avg_spin < opt['Spin Rate']:
                        recommendations.append(f"- {pitch}: Refine grip & wrist mechanics to increase spin.")

            else:
                tk.Label(content,
                         text="No optimal data defined.",
                         bg="#f1f2f6").pack(anchor="w")

            row_index += 1

        # =============================
        # RECOMMENDATIONS
        # =============================
        section_title("Recommendations to Reach Optimal Metrics")

        recommend_box = tk.Frame(metrics_frame, bg="#fef9e7", bd=2, relief="solid")
        recommend_box.pack(fill=tk.BOTH, expand=True, pady=15)

        if not recommendations:
            recommendations.append("- All pitch metrics meet or exceed optimal values.")

        for i, rec in enumerate(recommendations):
            tk.Label(recommend_box,
                     text=rec,
                     font=("Helvetica", 12),
                     bg="#fef9e7",
                     anchor="w",
                     justify="left",
                     wraplength=1100).grid(row=i, column=0, sticky="w", padx=25, pady=8)


# ===== RUN APPLICATION =====
root = tk.Tk()
app = TrackManAnalysis(root)
root.mainloop()
