import json
import h5py
from pathlib import Path

SRC = Path("hybrid_model.h5")
DST = Path("hybrid_model_fixed.h5")

if not SRC.exists():
    raise FileNotFoundError("hybrid_model.h5 not found next to this script")

# Copy file bytes first
DST.write_bytes(SRC.read_bytes())

with h5py.File(DST, "r+") as f:
    if "model_config" not in f.attrs:
        raise KeyError("No model_config found in the .h5 file attrs. This file may be weights-only.")

    raw = f.attrs["model_config"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    cfg = json.loads(raw)

    # Walk layers and fix InputLayer config key
    layers = cfg.get("config", {}).get("layers", [])
    changed = 0
    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            lcfg = layer.get("config", {})
            if "batch_shape" in lcfg and "batch_input_shape" not in lcfg:
                lcfg["batch_input_shape"] = lcfg.pop("batch_shape")
                layer["config"] = lcfg
                changed += 1

    # Save back
    f.attrs["model_config"] = json.dumps(cfg).encode("utf-8")

print(f"✅ Done. Created: {DST.name} | Fixed InputLayer entries: {changed}")
