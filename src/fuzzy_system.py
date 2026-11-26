# Mamdani fuzzy inference engine: CP05 covers core Mamdani inference, defuzzification, and plots;
# CP07 adds rule-firing trace output (infer_with_trace) to support plain-English explanations.

import os
import yaml
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# --- CP05: Membership function evaluation helper ---
def eval_mf(x: np.ndarray, params: list[float]) -> np.ndarray:
    """Evaluate triangular [a, b, c] or trapezoidal [a, b, c, d] membership on array x."""
    if len(params) == 3:
        return fuzz.trimf(x, params)
    if len(params) == 4:
        return fuzz.trapmf(x, params)
    raise ValueError(f"Expected 3 or 4 membership parameters, got {len(params)}: {params}")


class FuzzySentimentAnalyzer:
    """Mamdani fuzzy inference system mapping (POS, NEG, INT) -> (polarity, ambivalence)."""

    # --- CP05: Load universes, membership functions, and rules from config/rules.yaml ---
    def __init__(self, rules_path: str = "config/rules.yaml") -> None:
        with open(rules_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        uni = self.cfg["universes"]
        self.x_pos = np.linspace(uni["pos"][0], uni["pos"][1], 1001)
        self.x_neg = np.linspace(uni["neg"][0], uni["neg"][1], 1001)
        self.x_int = np.linspace(uni["int"][0], uni["int"][1], 1001)

        pol_pts = int(uni.get("polarity_points", 2001))
        amb_pts = int(uni.get("ambivalence_points", 1001))
        self.x_pol = np.linspace(uni["polarity"][0], uni["polarity"][1], pol_pts)
        self.x_amb = np.linspace(uni["ambivalence"][0], uni["ambivalence"][1], amb_pts)

        mfs = self.cfg["memberships"]
        self.pos_mfs = {k: eval_mf(self.x_pos, v) for k, v in mfs["POS"].items()}
        self.neg_mfs = {k: eval_mf(self.x_neg, v) for k, v in mfs["NEG"].items()}
        self.int_mfs = {k: eval_mf(self.x_int, v) for k, v in mfs["INT"].items()}
        self.pol_mfs = {k: eval_mf(self.x_pol, v) for k, v in mfs["POLARITY"].items()}
        self.amb_mfs = {k: eval_mf(self.x_amb, v) for k, v in mfs["AMBIVALENCE"].items()}

        self.rules = self.cfg["rules"]

    # --- CP05: Membership degree interpolation ---
    def _degree(self, x_grid: np.ndarray, mf_arr: np.ndarray, val: float) -> float:
        clipped_val = float(np.clip(val, x_grid[0], x_grid[-1]))
        return float(fuzz.interp_membership(x_grid, mf_arr, clipped_val))

    # --- CP07: Rule-firing trace extension for explanation generation ---
    def infer_with_trace(self, pos: float, neg: float, intensity: float) -> dict:
        """Run Mamdani inference and return polarity, ambivalence, and active rules."""
        pos_val = float(np.clip(pos, 0.0, 1.0))
        neg_val = float(np.clip(neg, 0.0, 1.0))
        int_val = float(np.clip(intensity, 0.0, 1.0))

        agg_pol = np.zeros_like(self.x_pol)
        agg_amb = np.zeros_like(self.x_amb)
        active_rules = []

        for rule in self.rules:
            firing_terms = []
            if rule.get("pos"):
                firing_terms.append(self._degree(self.x_pos, self.pos_mfs[rule["pos"]], pos_val))
            if rule.get("neg"):
                firing_terms.append(self._degree(self.x_neg, self.neg_mfs[rule["neg"]], neg_val))
            if rule.get("int"):
                firing_terms.append(self._degree(self.x_int, self.int_mfs[rule["int"]], int_val))

            strength = min(firing_terms) if firing_terms else 0.0
            if strength <= 0.0:
                continue

            # CP07: Record firing rule names and strengths
            active_rules.append(
                {
                    "name": rule.get("name", "unnamed_rule"),
                    "strength": round(float(strength), 4),
                    "polarity": rule.get("polarity"),
                    "ambivalence": rule.get("ambivalence"),
                }
            )

            if rule.get("polarity"):
                clipped_pol = np.minimum(strength, self.pol_mfs[rule["polarity"]])
                agg_pol = np.maximum(agg_pol, clipped_pol)

            if rule.get("ambivalence"):
                clipped_amb = np.minimum(strength, self.amb_mfs[rule["ambivalence"]])
                agg_amb = np.maximum(agg_amb, clipped_amb)

        # CP07: Sort active rules from strongest to weakest firing strength
        active_rules.sort(key=lambda r: r["strength"], reverse=True)

        if np.sum(agg_pol) > 0.0:
            polarity = float(fuzz.defuzz(self.x_pol, agg_pol, "centroid"))
        else:
            polarity = float(np.clip(pos_val - neg_val, -1.0, 1.0))

        if np.sum(agg_amb) > 0.0:
            ambivalence = float(fuzz.defuzz(self.x_amb, agg_amb, "centroid"))
        else:
            ambivalence = float(np.clip(min(pos_val, neg_val), 0.0, 1.0))

        return {
            "polarity": float(np.clip(polarity, -1.0, 1.0)),
            "ambivalence": float(np.clip(ambivalence, 0.0, 1.0)),
            "active_rules": active_rules,
        }

    # --- CP05: Single-sample inference ---
    def infer(self, pos: float, neg: float, intensity: float) -> dict[str, float]:
        """Run Mamdani inference and centroid defuzzification for a single (POS, NEG, INT) tuple."""
        traced = self.infer_with_trace(pos, neg, intensity)
        return {
            "polarity": traced["polarity"],
            "ambivalence": traced["ambivalence"],
        }

    # --- CP05: Batch inference over arrays ---
    def infer_batch(self, pos_arr, neg_arr, int_arr) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized wrapper over arrays of POS, NEG, and INT."""
        pol_out = []
        amb_out = []
        for p, n, i in zip(pos_arr, neg_arr, int_arr):
            res = self.infer(float(p), float(n), float(i))
            pol_out.append(res["polarity"])
            amb_out.append(res["ambivalence"])
        return np.asarray(pol_out, dtype=float), np.asarray(amb_out, dtype=float)

    # --- CP05: Membership function plot generation ---
    def plot_memberships(self, out_path: str = "results/membership.png") -> str:
        """Plot input (POS, NEG, INT) and output (POLARITY, AMBIVALENCE) membership functions."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes_flat = axes.flatten()

        panels = [
            (axes_flat[0], self.x_pos, self.pos_mfs, "Input: POS", "POS"),
            (axes_flat[1], self.x_neg, self.neg_mfs, "Input: NEG", "NEG"),
            (axes_flat[2], self.x_int, self.int_mfs, "Input: INT", "INT"),
            (axes_flat[3], self.x_pol, self.pol_mfs, "Output: POLARITY", "Polarity"),
            (axes_flat[4], self.x_amb, self.amb_mfs, "Output: AMBIVALENCE", "Ambivalence"),
        ]

        for ax, x_grid, mf_dict, title, xlabel in panels:
            for label, arr in mf_dict.items():
                ax.plot(x_grid, arr, linewidth=2, label=label)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Membership")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best", fontsize=8)

        axes_flat[5].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


# --- CP05: Standalone plot run ---
if __name__ == "__main__":
    analyzer = FuzzySentimentAnalyzer()
    saved = analyzer.plot_memberships("results/membership.png")
    print(f"Saved membership plot -> {saved}", flush=True)