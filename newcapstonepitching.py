import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===== XGBoost & ML imports =====
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np


class TrackManAnalysis:
    def __init__(self, root):
        self.root = root
        self.root.title("Trackman Pitching Analysis Development Program")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")

        header = tk.Frame(self.root, bg="#1f2c3c", pady=20)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Trackman Pitching Analysis Dashboard",
            fg="white",
            bg="#1f2c3c",
            font=("Helvetica", 22, "bold")
        ).pack()

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

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        metrics_frame = tk.Frame(self.scrollable_frame, bg="white")
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        def section_title(title):
            tk.Label(metrics_frame, text=title, font=("Helvetica", 18, "bold"),
                     bg="white").pack(anchor="w", pady=15)

        def metric_card(parent, title, value, row, column):
            card = tk.Frame(parent, bg="#ecf0f1", bd=2, relief="ridge")
            card.grid(row=row, column=column, padx=25, pady=20, sticky="nsew")
            tk.Label(card, text=title, font=("Helvetica", 11, "bold"),
                     bg="#ecf0f1").pack(padx=25, pady=(15, 5))
            tk.Label(card, text=value, font=("Helvetica", 16),
                     bg="#ecf0f1").pack(padx=25, pady=(0, 15))

        # =============================
        # BALL vs STRIKE MODEL
        # =============================
        section_title("XGBoost Ball vs Strike Probability Model")

        required_cols = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'description']
        if not all(col in df.columns for col in required_cols):
            tk.Label(metrics_frame, text="Dataset missing required columns for ML model.", bg="white").pack()
            return

        strike_events = ['called_strike','swinging_strike','swinging_strike_blocked','foul','foul_tip','hit_into_play']
        df['strike'] = df['description'].isin(strike_events).astype(int)

        features = ['release_speed','release_spin_rate','pfx_x','pfx_z','plate_x','plate_z']
        df_ml = df.dropna(subset=features + ['strike']).copy()

        X = df_ml[features]
        y = df_ml['strike']

        if y.nunique() <= 1 or len(df_ml) < 50:
            tk.Label(metrics_frame, text="Not enough data variation for model training.", bg="white").pack()
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                              subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        strike_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ball_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        perf_frame = tk.Frame(metrics_frame, bg="white")
        perf_frame.pack(anchor="w")

        metric_card(perf_frame, "Strike Accuracy", f"{accuracy*100:.1f}%", 0, 0)
        metric_card(perf_frame, "Strike AUC", f"{auc:.3f}", 0, 1)
        metric_card(perf_frame, "Strike Detection Rate", f"{strike_recall*100:.1f}%", 1, 0)
        metric_card(perf_frame, "Ball Detection Rate", f"{ball_recall*100:.1f}%", 1, 1)
        metric_card(perf_frame, "Strike Prediction Precision", f"{precision*100:.1f}%", 1, 2)

        fig2, ax3 = plt.subplots(figsize=(6, 4))
        ax3.barh(features, model.feature_importances_)
        ax3.set_title("Feature Importance (Strike Prediction)")

        chart_canvas2 = FigureCanvasTkAgg(fig2, master=self.scrollable_frame)
        chart_canvas2.draw()
        chart_canvas2.get_tk_widget().pack(pady=20)

        # ======================================================
        # SWING & MISS (WHIFF) PROBABILITY MODEL
        # ======================================================
        section_title("XGBoost Swing & Miss (Whiff) Probability Model")

        whiff_events = ['swinging_strike','swinging_strike_blocked']
        df['whiff'] = df['description'].isin(whiff_events).astype(int)
        df_whiff = df.dropna(subset=features + ['whiff']).copy()

        X_w = df_whiff[features]
        y_w = df_whiff['whiff']

        if y_w.nunique() > 1 and len(df_whiff) > 50:
            X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.25, random_state=42)
            whiff_model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                        subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
            whiff_model.fit(X_train_w, y_train_w)

            y_pred_w = whiff_model.predict(X_test_w)
            y_prob_w = whiff_model.predict_proba(X_test_w)[:, 1]

            whiff_accuracy = accuracy_score(y_test_w, y_pred_w)
            whiff_auc = roc_auc_score(y_test_w, y_prob_w)

            cm_w = confusion_matrix(y_test_w, y_pred_w)
            tn_w, fp_w, fn_w, tp_w = cm_w.ravel()

            whiff_recall = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0
            whiff_precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0

            whiff_frame = tk.Frame(metrics_frame, bg="white")
            whiff_frame.pack(anchor="w")

            metric_card(whiff_frame, "Whiff Accuracy", f"{whiff_accuracy*100:.1f}%", 0, 0)
            metric_card(whiff_frame, "Whiff AUC", f"{whiff_auc:.3f}", 0, 1)
            metric_card(whiff_frame, "Whiff Detection Rate", f"{whiff_recall*100:.1f}%", 1, 0)
            metric_card(whiff_frame, "Whiff Prediction Precision", f"{whiff_precision*100:.1f}%", 1, 1)

            fig_whiff, ax_whiff = plt.subplots(figsize=(6, 4))
            ax_whiff.barh(features, whiff_model.feature_importances_)
            ax_whiff.set_title("Feature Importance (Whiff Prediction)")

            canvas_whiff = FigureCanvasTkAgg(fig_whiff, master=self.scrollable_frame)
            canvas_whiff.draw()
            canvas_whiff.get_tk_widget().pack(pady=20)
        else:
            tk.Label(metrics_frame, text="Not enough variation to train Whiff model.", bg="white").pack()

        # ======================================================
        # PITCH TYPE STRIKE & WHIFF LEADERBOARD
        # ======================================================
        section_title("Pitch Arsenal Effectiveness Leaderboard")

        if 'pitch_type' in df.columns:

            df_rank = df.dropna(subset=features + ['pitch_type']).copy()
            df_rank['strike_prob'] = model.predict_proba(df_rank[features])[:, 1]
            df_rank['whiff_prob'] = whiff_model.predict_proba(df_rank[features])[:, 1]

            summary = df_rank.groupby('pitch_type').agg(
                Avg_Strike_Prob=('strike_prob', 'mean'),
                Avg_Whiff_Prob=('whiff_prob', 'mean'),
                Pitch_Count=('pitch_type', 'count')
            ).reset_index()

            summary = summary[summary['Pitch_Count'] >= 10]

            if len(summary) == 0:
                tk.Label(metrics_frame, text="Not enough pitch volume for ranking.", bg="white").pack()
                return

            summary = summary.sort_values(by=['Avg_Whiff_Prob', 'Avg_Strike_Prob'], ascending=False).reset_index(drop=True)
            best_strike_pitch = summary.loc[summary['Avg_Strike_Prob'].idxmax(), 'pitch_type']
            best_whiff_pitch = summary.loc[summary['Avg_Whiff_Prob'].idxmax(), 'pitch_type']

            # ===== Display leaderboard table =====
            table_frame = tk.Frame(metrics_frame, bg="white")
            table_frame.pack(pady=15)

            headers = ["Pitch Type", "Avg Strike %", "Avg Whiff %", "Count"]
            for col, header in enumerate(headers):
                tk.Label(table_frame, text=header, font=("Helvetica", 11, "bold"), bg="white",
                         borderwidth=1, relief="solid", padx=10, pady=5).grid(row=0, column=col, sticky="nsew")

            for row_idx, row in summary.iterrows():
                pitch = row['pitch_type']
                bg_color = "#ffffff"
                if pitch == best_whiff_pitch:
                    bg_color = "#d4f5d4"  # green for best whiff
                if pitch == best_strike_pitch:
                    bg_color = "#d9e8fb"  # blue for best strike

                tk.Label(table_frame, text=pitch, bg=bg_color, borderwidth=1,
                         relief="solid", padx=10, pady=5).grid(row=row_idx+1, column=0)
                tk.Label(table_frame, text=f"{row['Avg_Strike_Prob']*100:.1f}%", bg=bg_color,
                         borderwidth=1, relief="solid", padx=10, pady=5).grid(row=row_idx+1, column=1)
                tk.Label(table_frame, text=f"{row['Avg_Whiff_Prob']*100:.1f}%", bg=bg_color,
                         borderwidth=1, relief="solid", padx=10, pady=5).grid(row=row_idx+1, column=2)
                tk.Label(table_frame, text=int(row['Pitch_Count']), bg=bg_color,
                         borderwidth=1, relief="solid", padx=10, pady=5).grid(row=row_idx+1, column=3)

            # ===== Coaching Insight Summary =====
            insight_frame = tk.Frame(metrics_frame, bg="white")
            insight_frame.pack(anchor="w", pady=10)
            tk.Label(insight_frame, text=f"Best Strike Pitch: {best_strike_pitch}", font=("Helvetica", 12, "bold"), bg="white").pack(anchor="w")
            tk.Label(insight_frame, text=f"Best Swing & Miss Pitch: {best_whiff_pitch}", font=("Helvetica", 12, "bold"), bg="white").pack(anchor="w")

            # ===== Optional: Display pitch type legend =====
            section_title("Pitch Type Legend")
            pitch_legend = {
                "ST": "Four-Seam Fastball",
                "FF": "Four-Seam Fastball",
                "FT": "Two-Seam Fastball",
                "SL": "Slider",
                "CU": "Curveball",
                "CH": "Changeup",
                "KC": "Knuckle Curve",
                "SI": "Sinker",
                "FS": "Split-Finger Fastball",
                "SC": "Screwball",
                "FC": "Cutter"
            }

            legend_frame = tk.Frame(metrics_frame, bg="white")
            legend_frame.pack(anchor="w", pady=10)

            for i, (abbr, desc) in enumerate(pitch_legend.items()):
                tk.Label(legend_frame, text=f"{abbr}: {desc}", font=("Helvetica", 11), bg="white")\
                    .grid(row=i // 2, column=i % 2, sticky="w", padx=20, pady=2)

        else:
            tk.Label(metrics_frame, text="No pitch_type column found for arsenal ranking.", bg="white").pack()


# ===== RUN APP =====
root = tk.Tk()
app = TrackManAnalysis(root)
root.mainloop()
