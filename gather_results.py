"""
Collect evaluation scores into tables.

Subcommands
-----------
gather
    Aggregate CSV scores produced by run scripts.

    Output directory structure written by eval_v3_wrapper:
      {log_dir}/layer_{N}/{proj_name}/{mode}/{proj_train_data}{suffix}.csv
      {log_dir}/{proj_name}/{mode}/{proj_train_data}{suffix}.csv
        mode   : rejection | projection
        suffix : -phone    | -spk

    Usage:
      python gather_results.py gather results_dir [--layers 0 1 ... 12] [--save-dir path]
      python gather_results.py gather '/path/to/reproduce_layer*' [--layers 0 1 ... 12]

score
    Compute retention / purity / leakage / interference from a score directory
    whose subdirectories follow the pattern {MODEL}_{LABEL}_{MODE}/.
    Writes raw_score-v2_{mode}.csv files.

    Usage:
      python gather_results.py score score_dir [--train-data dev-clean]
                               [--seeds 0] [--models CLF LEACE] [--labels spk phone]
                               [--modes clean] [--save-dir path]
      python gather_results.py score '/path/to/reproduce_layer*' [--layers 0 1 ... 12]
"""

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd


_LAYER_DIR_RE = re.compile(r"^(?:reproduce_layer_?|layer_?)(\d+)$")


def _layer_id_from_path(path: str) -> int | None:
    m = _LAYER_DIR_RE.fullmatch(Path(path).name)
    return int(m.group(1)) if m else None


def _resolve_layer_dirs(
    base_output: str,
    layers: list[int] | None = None,
    required: bool = False,
) -> list[tuple[int, str]]:
    """Return sorted (layer_id, layer_dir) pairs for supported layer layouts."""
    has_magic = glob.has_magic(base_output)
    if has_magic:
        candidates = glob.glob(base_output)
        search_desc = base_output
    elif os.path.isdir(base_output) and _layer_id_from_path(base_output) is not None:
        candidates = [base_output]
        search_desc = base_output
    else:
        search_desc = os.path.join(base_output, "*")
        candidates = glob.glob(search_desc)

    valid = []
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        layer_id = _layer_id_from_path(candidate)
        if layer_id is None:
            continue
        valid.append((layer_id, candidate))

    if layers is not None:
        keep = set(layers)
        filtered = [(layer_id, path) for layer_id, path in valid if layer_id in keep]
    else:
        filtered = valid

    if filtered:
        return sorted(filtered, key=lambda item: (item[0], item[1]))

    if valid:
        raise ValueError(f"No layer directories matched --layers {layers}")

    if required or has_magic:
        if not candidates:
            raise FileNotFoundError(f"No paths matched '{search_desc}'")
        raise ValueError(
            f"No supported layer directories matched '{search_desc}'. "
            "Expected names like layer0, layer_0, reproduce_layer0, or reproduce_layer_0."
        )

    return []


def _default_score_save_dir(base_output: str, layer_dirs: list[tuple[int, str]]) -> str:
    if not glob.has_magic(base_output):
        return base_output
    parents = [os.path.dirname(os.path.abspath(path)) for _, path in layer_dirs]
    return os.path.commonpath(parents) if parents else os.curdir


# ---------------------------------------------------------------------------
# gather subcommand
# ---------------------------------------------------------------------------

def _read_score_csvs(score_dir: str, layer_id: int | None = None) -> list[pd.DataFrame]:
    frames = []
    csv_files = glob.glob(os.path.join(score_dir, "*", "*", "*.csv"))
    for csv_path in csv_files:
        parts = Path(csv_path).parts
        mode = parts[-2]
        stem = Path(csv_path).stem  # e.g. "dev-clean-phone" or "dev-clean-spk"

        if stem.endswith("-spk"):
            target = "spk"
        elif stem.endswith("-phone"):
            target = "phone"
        else:
            target = stem

        df = pd.read_csv(csv_path)
        if layer_id is not None:
            df["layer"] = layer_id
        df["mode"] = mode
        df["target"] = target
        frames.append(df)
    return frames


def gather(log_dir: str, layers: list[int] | None = None) -> pd.DataFrame:
    layer_dirs = _resolve_layer_dirs(log_dir, layers=layers, required=False)

    frames = []
    if layer_dirs:
        for layer_id, layer_dir in layer_dirs:
            frames.extend(_read_score_csvs(layer_dir, layer_id=layer_id))
    elif layers is None:
        frames.extend(_read_score_csvs(log_dir))

    if not frames:
        raise ValueError("No CSV files found under the given prefix/layers")

    result = pd.concat(frames, ignore_index=True)
    meta_cols = ["layer", "proj name", "mode", "target", "proj_train_data", "clf_train_data"]
    data_cols = [c for c in result.columns if c not in meta_cols]
    result = result[[c for c in meta_cols if c in result.columns] + data_cols]
    sort_cols = [c for c in ["layer", "proj name", "mode", "target"] if c in result.columns]
    result = result.sort_values(sort_cols).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# score subcommand
# ---------------------------------------------------------------------------

_KNOWN_MODES = {"clean", "other", "all"}
_KNOWN_LABELS = {"spk", "phone"}
_METHOD_SORT_ORDER = {
    "WORST": 0,
    "BEST": 1,
    "EYE": 2,
    "RANDOM": 3,
    "LC": 4,
    "LDA": 5,
    "PCA": 6,
    "COV": 7,
    "LEACE": 8,
}


def _discover(base_output: str):
    """Return (model_names, label_names, modes) discovered from subdirectory names."""
    model_names: set[str] = set()
    label_names: set[str] = set()
    found_modes: set[str] = set()

    for entry in os.scandir(base_output):
        if not entry.is_dir():
            continue
        name = entry.name
        for mode in _KNOWN_MODES:
            suffix = f"_{mode}"
            if not name.endswith(suffix):
                continue
            prefix = name[: -len(suffix)]
            found_modes.add(mode)
            if prefix == "EYE":
                model_names.add("EYE")
            elif prefix.startswith("ITER_"):
                model_names.add(prefix)
            else:
                for label in _KNOWN_LABELS:
                    if prefix.endswith(f"_{label}"):
                        model_names.add(prefix[: -len(f"_{label}")])
                        label_names.add(label)
                        break

    return sorted(model_names), sorted(label_names), sorted(found_modes)


def _subdir(base_output: str, model: str, label: str, mode: str) -> str:
    """Return the subdirectory path for a given (model, label, mode) triple."""
    if model == "EYE" or model.startswith("ITER_"):
        return os.path.join(base_output, f"{model}_{mode}")
    return os.path.join(base_output, f"{model}_{label}_{mode}")


def _score_path(subdir: str, split: str, label: str, train_data: str) -> str:
    return os.path.join(subdir, split, f"{train_data}-{label}.csv")


def _read_score_full(
    subdir: str,
    split: str,
    label: str,
    mode: str,
    mode_test: str,
    train_data: str,
) -> float:
    csv_path = _score_path(subdir, split, label, train_data)
    df = pd.read_csv(csv_path)
    rows = df[df["clf_train_data"] == f"dev-{mode}"][f"test-{mode_test}"].tolist()
    if len(rows) != 1:
        raise ValueError(f"Expected 1 row, got {len(rows)} in {csv_path!r} "
                         f"for clf_train_data=dev-{mode}, col=test-{mode_test}")
    return rows[0]


def gather_concept_scores(
    base_output: str,
    model_names: list[str] | None = None,
    label_names: list[str] | None = None,
    modes: list[str] | None = None,
    seeds: list[int] = (0,),
    train_data: str = "dev-clean",
) -> pd.DataFrame:
    disc_models, disc_labels, disc_modes = _discover(base_output)

    if model_names is None:
        model_names = disc_models
    if label_names is None:
        label_names = disc_labels
    if modes is None:
        modes = disc_modes

    records = []
    label_pairs = [tuple(label_names), tuple(reversed(label_names))]

    for mode in modes:
        mode_test = "clean" if mode == "all" else mode

        for model in model_names:
            for lab1, lab2 in label_pairs:
                for seed in seeds:
                    d = _subdir(base_output, model, lab1, mode)
                    paths = [
                        _score_path(d, "projection", lab1, train_data),
                        _score_path(d, "rejection",  lab1, train_data),
                        _score_path(d, "projection", lab2, train_data),
                        _score_path(d, "rejection",  lab2, train_data),
                    ]
                    if any(not os.path.exists(p) for p in paths):
                        missing = [p for p in paths if not os.path.exists(p)]
                        print(f"skipping {model}/{lab1}/{mode}: missing {missing}")
                        continue

                    retention    = _read_score_full(d, "projection", lab1, mode, mode_test, train_data)
                    leakage      = _read_score_full(d, "rejection",  lab1, mode, mode_test, train_data)
                    purity       = _read_score_full(d, "projection", lab2, mode, mode_test, train_data)
                    interference = _read_score_full(d, "rejection",  lab2, mode, mode_test, train_data)

                    records.append({
                        "concept": lab1,
                        "method": model,
                        "mode": mode,
                        "seed": seed,
                        "retention": retention,
                        "purity": 1 - purity,
                        "leakage": leakage,
                        "interference": 1 - interference,
                    })

    return pd.DataFrame(records).drop_duplicates()


def add_best_worst_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append notebook-style BEST/WORST rows derived from EYE rows."""
    required_cols = {"concept", "method", "mode", "seed", "retention", "purity", "leakage", "interference"}
    if df.empty or not required_cols.issubset(df.columns):
        return df

    group_cols = ["mode", "seed"]
    if "layer" in df.columns:
        group_cols.insert(0, "layer")

    synthetic_records = []
    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        eye = group[group["method"] == "EYE"]
        concepts = list(dict.fromkeys(eye["concept"].tolist()))
        if len(concepts) != 2:
            continue

        eye_by_concept = {row["concept"]: row for _, row in eye.iterrows()}
        if set(eye_by_concept) != set(concepts):
            continue

        for concept in concepts:
            other = concepts[1] if concept == concepts[0] else concepts[0]
            current = eye_by_concept[concept]
            paired = eye_by_concept[other]

            for method, values in {
                "BEST": {
                    "retention": current["retention"],
                    "purity": 1 - paired["leakage"],
                    "leakage": current["leakage"],
                    "interference": 1 - paired["retention"],
                },
                "WORST": {
                    "retention": current["leakage"],
                    "purity": 1 - paired["retention"],
                    "leakage": current["retention"],
                    "interference": 1 - paired["leakage"],
                },
            }.items():
                existing = group[(group["method"] == method) & (group["concept"] == concept)]
                if not existing.empty:
                    continue

                record = {col: pd.NA for col in df.columns}
                for col in group_cols:
                    record[col] = current[col]
                record.update({
                    "concept": concept,
                    "method": method,
                    **values,
                })
                synthetic_records.append(record)

    if not synthetic_records:
        return df

    synthetic = pd.DataFrame(synthetic_records, columns=df.columns)
    return pd.concat([df, synthetic], ignore_index=True)


def _score_sort_key(values: pd.Series) -> pd.Series:
    if values.name != "method":
        return values

    def method_key(value):
        method = str(value)
        if method in _METHOD_SORT_ORDER:
            return (_METHOD_SORT_ORDER[method], 0, method)

        random_match = re.fullmatch(r"RANDOM(\d+)", method)
        if random_match:
            return (_METHOD_SORT_ORDER["RANDOM"], int(random_match.group(1)), method)

        leace_comp_match = re.fullmatch(r"LEACE-comp(\d+|None)", method)
        if leace_comp_match:
            component = leace_comp_match.group(1)
            component_order = 10_000 if component == "None" else int(component)
            return (_METHOD_SORT_ORDER["LEACE"] + 1, component_order, method)

        return (1_000, 0, method)

    return values.map(method_key)


def sort_score_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "layer" in df.columns:
        sort_cols = [col for col in ["layer", "concept", "method", "mode", "seed"] if col in df.columns]
    else:
        sort_cols = [col for col in ["concept", "method", "mode", "seed"] if col in df.columns]
    if df.empty or not sort_cols:
        return df
    return df.sort_values(sort_cols, key=_score_sort_key, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_gather(args):
    result = gather(args.log_dir, layers=args.layers)
    print(result.to_string())

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "raw_gather.csv")
        result.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")


def _cmd_score(args):
    shared = dict(
        model_names=args.models or None,
        label_names=args.labels or None,
        modes=args.modes or None,
        seeds=args.seeds,
        train_data=args.train_data,
    )

    all_layer_dirs = _resolve_layer_dirs(
        args.base_output,
        required=glob.has_magic(args.base_output),
    )
    if args.layers is not None:
        keep = set(args.layers)
        layer_dirs = [(layer_id, path) for layer_id, path in all_layer_dirs if layer_id in keep]
        if all_layer_dirs and not layer_dirs:
            raise ValueError(f"No layer directories matched --layers {args.layers}")
    else:
        layer_dirs = all_layer_dirs

    if all_layer_dirs:
        frames = []
        for layer_id, layer_dir in layer_dirs:
            df = gather_concept_scores(base_output=layer_dir, **shared)
            df.insert(0, "layer", layer_id)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = gather_concept_scores(base_output=args.base_output, **shared)

    df = add_best_worst_rows(df)
    df = sort_score_rows(df)
    print(df.to_string())

    if df.empty:
        return

    save_dir = args.save_dir or _default_score_save_dir(args.base_output, layer_dirs)
    os.makedirs(save_dir, exist_ok=True)
    for mode in df["mode"].unique():
        out_path = os.path.join(save_dir, f"raw_score-v2_{mode}.csv")
        df[df["mode"] == mode].to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather evaluation scores.")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- gather ---
    p_gather = sub.add_parser("gather", help="Aggregate layer-wise scores from run_acl_test_suite output.")
    p_gather.add_argument("log_dir", help="log_dir passed to run_acl_test_suite.py, or a quoted layer glob")
    p_gather.add_argument("--layers", nargs="+", type=int, default=None,
                          help="Layer IDs to include (default: all found)")
    p_gather.add_argument("--save-dir", default=None,
                          help="Directory to write raw_gather.csv")

    # --- score ---
    p_score = sub.add_parser("score", help="Compute retention/purity/leakage/interference metrics.")
    p_score.add_argument("base_output", help="Directory containing {MODEL}_{LABEL}_{MODE}/ subdirs, or a quoted layer glob")
    p_score.add_argument("--train-data", default="dev-clean",
                         help="Train data prefix used in score filenames (default: dev-clean)")
    p_score.add_argument("--seeds", nargs="+", type=int, default=[0],
                         help="Seed values to record (default: 0)")
    p_score.add_argument("--models", nargs="+", default=None,
                         help="Model names to include (default: auto-discover)")
    p_score.add_argument("--labels", nargs="+", default=None,
                         help="Label names to include (default: auto-discover)")
    p_score.add_argument("--modes", nargs="+", default=None,
                         help="Modes to include (default: auto-discover)")
    p_score.add_argument("--layers", nargs="+", type=int, default=None,
                         help="Layer IDs to include when base_output contains layer_*/ subdirs (default: all found)")
    p_score.add_argument("--save-dir", default=None,
                         help="Directory to write raw_score-v2_{mode}.csv (default: base_output)")

    args = parser.parse_args()
    if args.command == "gather":
        _cmd_gather(args)
    else:
        _cmd_score(args)
