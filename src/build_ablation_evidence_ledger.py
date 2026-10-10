from pathlib import Path
import csv
import json
import re
import torch


ROOT = Path("/root/autodl-tmp/CVIP2026_DCEC_v11_JOURNAL")

OUT_DIR = ROOT / "results/provenance"

OUT_JSON = OUT_DIR / "ABLATION_EVIDENCE_LEDGER_20260926.json"
OUT_TXT = OUT_DIR / "ABLATION_EVIDENCE_LEDGER_20260926.txt"
OUT_CSV = OUT_DIR / "ABLATION_CONTROLLED_SELECTION_20260926.csv"


CONTROLLED = [
    {
        "label": "A1",
        "variant": "residual_only",
        "params": 262912,
        "checkpoint":
            ROOT / "results/ablation_formal/residual_only/best_dcec_ablation.pt",
    },
    {
        "label": "A2",
        "variant": "quality_no_gate",
        "params": 281600,
        "checkpoint":
            ROOT / "results/ablation_formal/quality_no_gate/best_dcec_ablation.pt",
    },
    {
        "label": "A3",
        "variant": "no_sem",
        "params": 283713,
        "checkpoint":
            ROOT / "results/ablation_formal/no_sem/best_dcec_ablation.pt",
    },
    {
        "label": "A4",
        "variant": "no_rank",
        "params": 283713,
        "checkpoint":
            ROOT / "results/ablation_formal/no_rank/best_dcec_ablation.pt",
    },
]


SUMMARY_FILE = (
    ROOT / "results/ablation_fullval/a3_vs_a5_summary.txt"
)


def read_checkpoint(row):
    path = row["checkpoint"]

    if not path.exists():
        return {
            **row,
            "exists": False,
        }

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return {
        "label": row["label"],
        "variant": row["variant"],
        "params": row["params"],
        "checkpoint": str(path),
        "exists": True,
        "checkpoint_variant": obj.get("variant"),
        "epoch": obj.get("epoch"),
        "seed": obj.get("seed"),
        "val_primary_score": obj.get("val_primary_score"),
    }


def parse_a3_a5_summary(path):
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    condition_pattern = re.compile(
        r"^(clean_S0|jpeg_S[123]|gaussian_blur_S[123]|"
        r"gaussian_noise_S[123]|low_light_S[123])\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([+-][0-9.]+)",
        re.MULTILINE,
    )

    conditions = []

    for match in condition_pattern.finditer(text):
        conditions.append(
            {
                "condition": match.group(1),
                "A3": float(match.group(2)),
                "A5": float(match.group(3)),
                "delta_A3_minus_A5": float(match.group(4)),
            }
        )

    m13 = re.search(
        r"13-condition mean:\s*"
        r"A3=([0-9.]+)\s+"
        r"A5=([0-9.]+)\s+"
        r"Delta=([+-][0-9.]+)",
        text,
    )

    m12 = re.search(
        r"12-corruption mean:\s*"
        r"A3=([0-9.]+)\s+"
        r"A5=([0-9.]+)\s+"
        r"Delta=([+-][0-9.]+)",
        text,
    )

    mwtl = re.search(
        r"wins=(\d+),\s*ties=(\d+),\s*losses=(\d+)",
        text,
    )

    result = {
        "exists": True,
        "path": str(path),
        "conditions": conditions,
    }

    if m13:
        result["mean_13_condition"] = {
            "A3": float(m13.group(1)),
            "A5": float(m13.group(2)),
            "delta_A3_minus_A5": float(m13.group(3)),
        }

    if m12:
        result["mean_12_corruption"] = {
            "A3": float(m12.group(1)),
            "A5": float(m12.group(2)),
            "delta_A3_minus_A5": float(m12.group(3)),
        }

    if mwtl:
        result["wins_ties_losses_A3_vs_A5"] = {
            "wins": int(mwtl.group(1)),
            "ties": int(mwtl.group(2)),
            "losses": int(mwtl.group(3)),
        }

    return result


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    controlled = [
        read_checkpoint(row)
        for row in CONTROLLED
    ]

    scores = [
        row["val_primary_score"]
        for row in controlled
        if row.get("val_primary_score") is not None
    ]

    best_score = max(scores)

    for row in controlled:
        score = row.get("val_primary_score")

        if score is not None:
            row["delta_vs_best_controlled"] = (
                float(score) - float(best_score)
            )

    a3_a5 = parse_a3_a5_summary(
        SUMMARY_FILE
    )

    ledger = {
        "title":
            "DCEC Ablation Evidence Ledger",

        "date":
            "2026-09-26",

        "controlled_selection_protocol_note":
            (
                "A1-A4 values are read directly from formal "
                "best checkpoints and share the controlled "
                "ablation selection protocol."
            ),

        "controlled_selection": controlled,

        "a3_vs_a5_full_13_condition_validation": a3_a5,

        "A0_frozen": {
            "status":
                "UNRESOLVED_PROVENANCE",

            "note":
                (
                    "A same-protocol controlled-selection source "
                    "for A0 Frozen CLIP has not yet been recovered. "
                    "Do not mix historical or differently scoped "
                    "Frozen CLIP numbers into the controlled table."
                ),
        },

        "A5_seed2026_checkpoint": {
            "status":
                "UNRESOLVED_PROVENANCE",

            "note":
                (
                    "The complete A3-vs-A5 13-condition validation "
                    "summary exists, but the exact seed2026 A5 "
                    "checkpoint path has not yet been recovered."
                ),
        },

        "interpretation_guardrails": [
            (
                "A3 vs A5 mean difference is small and should "
                "not be described as a large or statistically "
                "significant superiority without additional evidence."
            ),
            (
                "A1-A4 controlled-selection scores and the A3-vs-A5 "
                "13-condition means are different evaluation scopes "
                "and must not be placed in one numerical column."
            ),
            (
                "A0 should remain unresolved in the controlled table "
                "until exact provenance is recovered."
            ),
        ],
    }

    OUT_JSON.write_text(
        json.dumps(
            ledger,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "label",
                "variant",
                "params",
                "best_epoch",
                "seed",
                "val_primary_score",
                "delta_vs_best_controlled",
                "checkpoint",
            ]
        )

        for row in controlled:
            writer.writerow(
                [
                    row.get("label"),
                    row.get("variant"),
                    row.get("params"),
                    row.get("epoch"),
                    row.get("seed"),
                    row.get("val_primary_score"),
                    row.get("delta_vs_best_controlled"),
                    row.get("checkpoint"),
                ]
            )

    lines = []

    lines.append(
        "DCEC ABLATION EVIDENCE LEDGER — 2026-09-26"
    )
    lines.append("=" * 100)
    lines.append("")

    lines.append(
        "TABLE A — CONTROLLED ABLATION SELECTION"
    )
    lines.append("-" * 100)

    lines.append(
        f"{'Variant':22s}"
        f"{'Params':>12s}"
        f"{'Epoch':>8s}"
        f"{'Seed':>8s}"
        f"{'Val Primary':>16s}"
        f"{'Δ vs A3':>14s}"
    )

    a3_score = None

    for row in controlled:
        if row["label"] == "A3":
            a3_score = float(
                row["val_primary_score"]
            )

    for row in controlled:
        score = float(
            row["val_primary_score"]
        )

        delta_a3 = (
            score - a3_score
        )

        name = (
            f"{row['label']} {row['variant']}"
        )

        lines.append(
            f"{name:22s}"
            f"{row['params']:12d}"
            f"{row['epoch']:8d}"
            f"{row['seed']:8d}"
            f"{score:16.6f}"
            f"{delta_a3:+14.6f}"
        )

    lines.append("")
    lines.append(
        "IMPORTANT: A0 Frozen CLIP is intentionally omitted "
        "from Table A because its exact same-protocol source "
        "has not yet been recovered."
    )

    lines.append("")
    lines.append(
        "TABLE B — A3 VS A5 FULL 13-CONDITION VALIDATION"
    )
    lines.append("-" * 100)

    if a3_a5.get("exists"):
        for row in a3_a5.get(
            "conditions",
            []
        ):
            lines.append(
                f"{row['condition']:24s}"
                f"A3={row['A3']:8.4f}  "
                f"A5={row['A5']:8.4f}  "
                f"Δ={row['delta_A3_minus_A5']:+8.4f}"
            )

        lines.append("")

        if "mean_13_condition" in a3_a5:
            row = a3_a5[
                "mean_13_condition"
            ]

            lines.append(
                "13-condition mean: "
                f"A3={row['A3']:.4f} "
                f"A5={row['A5']:.4f} "
                f"Δ={row['delta_A3_minus_A5']:+.4f}"
            )

        if "mean_12_corruption" in a3_a5:
            row = a3_a5[
                "mean_12_corruption"
            ]

            lines.append(
                "12-corruption mean: "
                f"A3={row['A3']:.4f} "
                f"A5={row['A5']:.4f} "
                f"Δ={row['delta_A3_minus_A5']:+.4f}"
            )

        if (
            "wins_ties_losses_A3_vs_A5"
            in a3_a5
        ):
            row = a3_a5[
                "wins_ties_losses_A3_vs_A5"
            ]

            lines.append(
                "A3 vs A5 W/T/L: "
                f"{row['wins']}/"
                f"{row['ties']}/"
                f"{row['losses']}"
            )

    lines.append("")
    lines.append(
        "PROVENANCE GAPS"
    )
    lines.append("-" * 100)

    lines.append(
        "1. A0 Frozen CLIP same-protocol controlled-selection source: UNRESOLVED"
    )

    lines.append(
        "2. A5 seed2026 exact checkpoint path: UNRESOLVED"
    )

    lines.append("")
    lines.append(
        "INTERPRETATION"
    )
    lines.append("-" * 100)

    lines.append(
        "A3 is the strongest of the recovered A1-A4 "
        "controlled ablations."
    )

    lines.append(
        "A3 exceeds A2 quality_no_gate by approximately "
        f"{a3_score - controlled[1]['val_primary_score']:+.4f}."
    )

    lines.append(
        "A3 exceeds A1 residual_only by approximately "
        f"{a3_score - controlled[0]['val_primary_score']:+.4f}."
    )

    lines.append(
        "A3 exceeds A4 no_rank by approximately "
        f"{a3_score - controlled[3]['val_primary_score']:+.4f}."
    )

    if (
        a3_a5.get("exists")
        and "mean_13_condition" in a3_a5
    ):
        row = a3_a5["mean_13_condition"]

        lines.append(
            "On full 13-condition validation, A3 and A5 "
            "are very close: "
            f"A3={row['A3']:.4f}, "
            f"A5={row['A5']:.4f}, "
            f"Δ={row['delta_A3_minus_A5']:+.4f}."
        )

    lines.append("")
    lines.append(
        "Do not merge Table A and Table B into one score column: "
        "they use different evaluation scopes."
    )

    OUT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("ABLATION EVIDENCE LEDGER COMPLETE")
    print("=" * 100)
    print()

    for row in controlled:
        print(
            row["label"],
            row["variant"],
            "params=",
            row["params"],
            "epoch=",
            row["epoch"],
            "score=",
            f"{row['val_primary_score']:.6f}",
        )

    print()

    if (
        a3_a5.get("exists")
        and "mean_13_condition" in a3_a5
    ):
        row = a3_a5["mean_13_condition"]

        print(
            "A3 vs A5 13-condition:",
            f"{row['A3']:.4f}",
            "vs",
            f"{row['A5']:.4f}",
            "delta=",
            f"{row['delta_A3_minus_A5']:+.4f}",
        )

    print()
    print(
        "A0 provenance:",
        "UNRESOLVED"
    )
    print(
        "A5 seed2026 checkpoint provenance:",
        "UNRESOLVED"
    )

    print()
    print("Saved:", OUT_JSON)
    print("Saved:", OUT_TXT)
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
