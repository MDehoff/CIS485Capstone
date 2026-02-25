import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===== XGBoost & ML imports =====
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

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

        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.plot_dashboard(df)
            except Exception as error:
                messagebox.showerror("Error", f"Could not read CSV: {error}")

    def plot_dashboard(self, df):
        required_cols = ['pitch_name', 'release_speed']
        if not all(col in df.columns for col in required_cols):
            messagebox.showwarning("Warning", "CSV must contain 'pitch_name' and 'release_speed'.")
            return

        df = df.dropna(subset=required_cols)

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # =============================
        # PITCH USAGE + VELOCITY
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
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        plt.setp(ax2.get_xticklabels(), rotation=30)

        chart_canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(pady=25)

        # =============================
        # METRICS FRAME
        # =============================
        metrics_frame = tk.Frame(self.scrollable_frame, bg="white")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        def section_title(title):
            tk.Label(metrics_frame, text=title, font=("Helvetica", 18, "bold"), bg="white").pack(anchor="w", pady=15)

        def metric_card(parent, title, value, row, column):
            card = tk.Frame(parent, bg="#ecf0f1", bd=2, relief="ridge")
            card.grid(row=row, column=column, padx=25, pady=20, sticky="nsew")
            tk.Label(card, text=title, font=("Helvetica", 11, "bold"), bg="#ecf0f1").pack(padx=25, pady=(15, 5))
            tk.Label(card, text=value, font=("Helvetica", 16), bg="#ecf0f1").pack(padx=25, pady=(0, 15))

        # =============================
        # PERFORMANCE
        # =============================
        section_title("Performance Metrics")
        perf_frame = tk.Frame(metrics_frame, bg="white")
        perf_frame.pack(anchor="w")
        metric_card(perf_frame, "Avg Velocity", f"{df['release_speed'].mean():.2f} MPH", 0, 0)

        # =============================
        # XGBOOST STRIKE MODEL
        # =============================
        section_title("Advanced XGBoost Strike Probability Model")
        if 'description' in df.columns:
            strike_events = [
                'called_strike', 'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ]
            df['strike'] = df['description'].isin(strike_events).astype(int)
            ml_features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z']

            if all(col in df.columns for col in ml_features + ['strike']):
                df_ml = df.dropna(subset=ml_features + ['strike']).copy()
                X = df_ml[ml_features]
                y = df_ml['strike']

                if y.nunique() > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=42
                    )
                    model = XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=42,
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob)

                    metric_card(perf_frame, "ML Strike Accuracy", f"{accuracy*100:.1f}%", 0, 1)
                    metric_card(perf_frame, "ML Strike AUC", f"{auc:.3f}", 0, 2)

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                    ax_cm.imshow(cm)
                    ax_cm.set_title("Confusion Matrix")
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    for i in range(2):
                        for j in range(2):
                            ax_cm.text(j, i, cm[i, j], ha="center", va="center")
                    chart_canvas_cm = FigureCanvasTkAgg(fig_cm, master=self.scrollable_frame)
                    chart_canvas_cm.draw()
                    chart_canvas_cm.get_tk_widget().pack(pady=20)

                    # Feature Importance
                    fig2, ax3 = plt.subplots(figsize=(6, 4))
                    ax3.barh(ml_features, model.feature_importances_)
                    ax3.set_title("Feature Importance (Strike Model)")
                    chart_canvas2 = FigureCanvasTkAgg(fig2, master=self.scrollable_frame)
                    chart_canvas2.draw()
                    chart_canvas2.get_tk_widget().pack(pady=20)

        # =============================
        # OPTIMAL METRICS
        # =============================
        section_title("Optimal Pitching Metrics Comparison")
        optimal_metrics = {
            "4-Seam Fastball": {"Velocity": 94, "Spin Rate": 2300},
            "Fastball": {"Velocity": 94, "Spin Rate": 2300},
            "Sinker": {"Velocity": 92, "Spin Rate": 2100},
            "Cutter": {"Velocity": 88, "Spin Rate": 2400},
            "Slider": {"Velocity": 85, "Spin Rate": 2500},
            "Sweeper": {"Velocity": 83, "Spin Rate": 2600},
            "Curveball": {"Velocity": 78, "Spin Rate": 2600},
            "Changeup": {"Velocity": 84, "Spin Rate": 2000}
        }

        recommendations = []
        for pitch in df['pitch_name'].unique():
            df_pitch = df[df['pitch_name'] == pitch]
            avg_vel = df_pitch['release_speed'].mean()
            avg_spin = df_pitch['release_spin_rate'].mean() if 'release_spin_rate' in df.columns else None

            box = tk.Frame(metrics_frame, bg="#f1f2f6", bd=2, relief="solid")
            box.pack(fill=tk.X, pady=10)
            tk.Label(box, text=pitch, font=("Helvetica", 14, "bold"), bg="#dcdde1").pack(fill=tk.X)
            content = tk.Frame(box, bg="#f1f2f6")
            content.pack(padx=20, pady=10)

            opt = optimal_metrics.get(pitch)
            if opt:
                tk.Label(content, text=f"Velocity: {avg_vel:.1f} MPH | Optimal: {opt['Velocity']} MPH", bg="#f1f2f6").pack(anchor="w")
                if avg_vel < opt['Velocity']:
                    recommendations.append(f"- {pitch}: Increase lower-body power to improve velocity.")
                if avg_spin:
                    tk.Label(content, text=f"Spin Rate: {avg_spin:.0f} RPM | Optimal: {opt['Spin Rate']} RPM", bg="#f1f2f6").pack(anchor="w")
                    if avg_spin < opt['Spin Rate']:
                        recommendations.append(f"- {pitch}: Improve grip & wrist mechanics to increase spin.")
            else:
                tk.Label(content, text="No optimal benchmark defined.", bg="#f1f2f6").pack(anchor="w")

        # =============================
        # RECOMMENDATIONS
        # =============================
        section_title("Development Recommendations")
        rec_box = tk.Frame(metrics_frame, bg="#fef9e7", bd=2, relief="solid")
        rec_box.pack(fill=tk.BOTH, expand=True, pady=10)

        if not recommendations:
            recommendations.append("- All metrics meet or exceed optimal benchmarks.")

        for rec in recommendations:
            tk.Label(rec_box, text=rec, bg="#fef9e7", font=("Helvetica", 12), anchor="w", justify="left").pack(anchor="w", padx=20, pady=5)

# ===== RUN APP =====
root = tk.Tk()
app = TrackManAnalysis(root)
root.mainloop()
