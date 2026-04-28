"""
wire_calibrator.py
==================
Patches generate_madrid2026.py and courtiq_engine.py to apply the
probability calibrator after the RF model prediction step.

Run once from your project root:
    python wire_calibrator.py
"""

import re
import sys
from pathlib import Path

def patch_generate_script(path: str) -> bool:
    """Patch generate_madrid2026.py (and any similar generate scripts)."""
    p = Path(path)
    if not p.exists():
        print(f"  SKIP: {path} not found")
        return False

    content = p.read_text(encoding="utf-8")

    # Already patched?
    if "prob_calibrator" in content:
        print(f"  SKIP: {path} already has calibrator")
        return False

    # 1. Add calibrator load after model loads
    # Find the block that ends with "Model loaded"
    old_model_load = 'print(f"  Loaded"); break'
    if old_model_load not in content:
        old_model_load = 'print("  Model loaded"); break'
    if old_model_load not in content:
        old_model_load = 'print("  Loaded")\n            break'

    new_model_load = old_model_load + """
    # Load probability calibrator
    cal_path = Path("./models/prob_calibrator.joblib")
    try:
        calibrator = load(cal_path) if cal_path.exists() else None
        if calibrator:
            print(f"  Calibrator loaded: {cal_path}")
        else:
            print("  No calibrator found — using raw probabilities")
    except Exception as e:
        calibrator = None
        print(f"  Calibrator load failed: {e}")"""

    # If the exact string isn't found, try a broader search
    if old_model_load not in content:
        # Find where the model loading loop ends and insert after
        idx = content.find("if not pipe:")
        if idx < 0:
            print(f"  ERROR: Could not find insertion point in {path}")
            return False
        insert_after = content.rfind("\n", 0, idx)
        cal_block = """
    # Load probability calibrator
    cal_path = Path("./models/prob_calibrator.joblib")
    try:
        calibrator = load(cal_path) if cal_path.exists() else None
        if calibrator:
            print(f"  Calibrator loaded: {cal_path}")
        else:
            print("  No calibrator found — using raw probabilities")
    except Exception:
        calibrator = None
"""
        content = content[:insert_after] + cal_block + content[insert_after:]
    else:
        content = content.replace(old_model_load, new_model_load, 1)

    # 2. Apply calibrator to raw model probability
    # Find the line that produces p_std from the model
    old_pred = "p_std=float(pipe.predict_proba(fd[fcols])[:,1][0]) if pipe else elo_p(sa[\"pre_elo\"],sb[\"pre_elo\"])"
    new_pred = """p_raw=float(pipe.predict_proba(fd[fcols])[:,1][0]) if pipe else elo_p(sa["pre_elo"],sb["pre_elo"])
            p_std=float(calibrator.predict([p_raw])[0]) if calibrator else p_raw"""

    if old_pred in content:
        content = content.replace(old_pred, new_pred, 1)
        print(f"  Patched model prediction line in {path}")
    else:
        # Try variant
        old_pred2 = 'p_std = float(pipe.predict_proba(feat_df[feature_cols])[:, 1][0])'
        new_pred2 = '''p_raw = float(pipe.predict_proba(feat_df[feature_cols])[:, 1][0])
            p_std = float(calibrator.predict([p_raw])[0]) if calibrator else p_raw'''
        if old_pred2 in content:
            content = content.replace(old_pred2, new_pred2, 1)
            print(f"  Patched model prediction line (variant) in {path}")
        else:
            print(f"  WARNING: Could not patch prediction line in {path} — may need manual edit")

    p.write_text(content, encoding="utf-8")
    print(f"  Patched: {path}")
    return True


def patch_engine(path: str = "courtiq_engine.py") -> bool:
    """
    Patch courtiq_engine.py to load and apply the calibrator
    inside the prediction pipeline (cmd_predict / build_predictions).
    """
    p = Path(path)
    if not p.exists():
        print(f"  SKIP: {path} not found")
        return False

    content = p.read_text(encoding="utf-8")

    if "prob_calibrator" in content:
        print(f"  SKIP: {path} already has calibrator")
        return False

    # Find the global model loading section
    # Look for where pipe and feature_cols are loaded
    model_load_marker = "bundle = load(model_path)"
    if model_load_marker not in content:
        model_load_marker = 'bundle = load(path)'

    if model_load_marker not in content:
        print(f"  ERROR: Could not find model load in {path}")
        return False

    # Find the last occurrence (the actual runtime load, not in a def)
    idx = content.rfind(model_load_marker)
    # Find the end of this block (next blank line or next function)
    block_end = content.find("\n\n", idx)
    if block_end < 0:
        block_end = idx + 200

    # Insert calibrator loading after the model load block
    cal_snippet = """
    # Load probability calibrator
    _cal_path = MODELS_DIR / "prob_calibrator.joblib"
    try:
        calibrator = load(_cal_path) if _cal_path.exists() else None
        if calibrator:
            print(f"  [model] Calibrator loaded")
    except Exception:
        calibrator = None
"""
    content = content[:block_end] + cal_snippet + content[block_end:]

    # Patch the prediction probability line in the engine
    # The engine computes p_std via pipe.predict_proba
    old_engine_pred = 'p_a_std = float(pipe.predict_proba(feat_df[feature_cols])[:, 1][0])'
    new_engine_pred = '''p_raw_std = float(pipe.predict_proba(feat_df[feature_cols])[:, 1][0])
            p_a_std = float(calibrator.predict([p_raw_std])[0]) if calibrator else p_raw_std'''

    if old_engine_pred in content:
        content = content.replace(old_engine_pred, new_engine_pred, 1)
        print(f"  Patched engine prediction line")
    else:
        print(f"  WARNING: Could not patch engine prediction line — may need manual edit")
        print(f"  Look for: pipe.predict_proba and wrap with calibrator.predict()")

    p.write_text(content, encoding="utf-8")
    print(f"  Patched: {path}")
    return True


def verify_syntax(path: str) -> bool:
    import ast
    try:
        ast.parse(Path(path).read_text(encoding="utf-8"))
        print(f"  Syntax OK: {path}")
        return True
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {path}: {e}")
        return False


if __name__ == "__main__":
    print("\n=== Wiring probability calibrator ===\n")

    targets = [
        "generate_madrid2026.py",
        "generate_barcelona2026.py",
        "generate_munich2026.py",
        "courtiq_engine.py",
    ]

    patched = 0
    for t in targets:
        if Path(t).exists():
            print(f"Patching {t}...")
            if "generate_" in t:
                ok = patch_generate_script(t)
            else:
                ok = patch_engine(t)
            if ok:
                verify_syntax(t)
                patched += 1
        else:
            print(f"Skipping {t} (not found)")

    print(f"\nDone. {patched} file(s) patched.")
    print("\nNext steps:")
    print("  1. git add generate_madrid2026.py courtiq_engine.py models/prob_calibrator.joblib")
    print("  2. git commit -m 'feat: apply probability calibrator to all predictions'")
    print("  3. python generate_madrid2026.py  ← regenerate with calibrated probs")
    print("  4. python courtiq_engine.py site --output docs/index.html")
