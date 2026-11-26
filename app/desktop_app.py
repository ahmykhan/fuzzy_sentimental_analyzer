# Checkpoint 9 (CP09) — Normal desktop application (tkinter) + optional batch CSV CLI mode

import argparse
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
from src.pipeline import analyse, analyse_batch_df, validate_text_input


def run_gui() -> None:
    root = tk.Tk()
    root.title("Fuzzy Sentiment Analyser")
    root.geometry("820x620")

    main = ttk.Frame(root, padding=14)
    main.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main, text="Enter English text (max 2,000 chars):").pack(anchor=tk.W)
    input_box = scrolledtext.ScrolledText(main, height=5, wrap=tk.WORD)
    input_box.pack(fill=tk.X, pady=(4, 8))

    btn_row = ttk.Frame(main)
    btn_row.pack(fill=tk.X, pady=(0, 10))

    status_var = tk.StringVar(value="Ready")

    def on_analyse() -> None:
        text = input_box.get("1.0", tk.END).strip()
        err = validate_text_input(text, max_chars=2000)
        if err:
            messagebox.showwarning("Input Check", err)
            return

        status_var.set("Running analysis...")
        root.update_idletasks()
        try:
            res = analyse(text)
        except Exception as exc:
            status_var.set("Error")
            messagebox.showerror("Analysis Error", str(exc))
            return

        summary_lines = [
            f"Verdict: {res['verdict'].title()}   |   Polarity: {res['polarity']:+.2f}   |   Ambivalence: {res['ambivalence']:.2f}   |   Mixed: {'Yes' if res['is_mixed'] else 'No'}",
            f"Fused Inputs (w={res['w']}): POS={res['features']['POS']}, NEG={res['features']['NEG']}, INT={res['features']['INT']}",
            f"Explanation (Flesch {res['flesch_score']}): {res['explanation']}",
            "",
            "Fired Rules:",
        ]
        if res["active_rules"]:
            for r in res["active_rules"]:
                summary_lines.append(f"  - {r['name']} (strength={r['strength']}, polarity={r['polarity']}, ambivalence={r['ambivalence']})")
        else:
            summary_lines.append("  - None above zero; fallback balance used.")

        output_box.config(state=tk.NORMAL)
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, "\n".join(summary_lines))
        output_box.config(state=tk.DISABLED)
        status_var.set(f"Done in {res['latency_s']}s")

    def on_batch_csv() -> None:
        in_path = filedialog.askopenfilename(
            title="Select CSV with a 'text' column (max 1,000 rows)",
            filetypes=[("CSV files", "*.csv")],
        )
        if not in_path:
            return
        try:
            df = pd.read_csv(in_path)
            out_df = analyse_batch_df(df, max_rows=1000)
            save_path = filedialog.asksaveasfilename(
                title="Save scored CSV",
                defaultextension=".csv",
                initialfile="fuzzy_sentiment_results.csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if save_path:
                out_df.to_csv(save_path, index=False)
                messagebox.showinfo("Batch Complete", f"Saved {len(out_df)} rows to:\n{save_path}")
        except Exception as exc:
            messagebox.showerror("Batch CSV Error", str(exc))

    ttk.Button(btn_row, text="Analyse Text", command=on_analyse).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row, text="Process Batch CSV...", command=on_batch_csv).pack(side=tk.LEFT)
    ttk.Label(btn_row, textvariable=status_var).pack(side=tk.RIGHT)

    ttk.Label(main, text="Result, Explanation & Fired Rules:").pack(anchor=tk.W)
    output_box = scrolledtext.ScrolledText(main, height=18, wrap=tk.WORD, state=tk.DISABLED)
    output_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    root.mainloop()


def run_cli_csv(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    out_df = analyse_batch_df(df, max_rows=1000)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzzy Sentiment Analyser Desktop App")
    parser.add_argument("--csv", type=str, default=None, help="Optional input CSV path with a 'text' column")
    parser.add_argument("--out", type=str, default="results/batch_output.csv", help="Output CSV path for --csv mode")
    args = parser.parse_args()

    if args.csv:
        run_cli_csv(args.csv, args.out)
    else:
        run_gui()