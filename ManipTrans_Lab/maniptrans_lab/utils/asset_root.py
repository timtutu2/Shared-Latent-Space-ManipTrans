import os
from pathlib import Path


# Preferred: an `assets/` dir inside the maniptrans_lab package (drop-in from
# the original ManipTrans layout). If that doesn't exist, fall back to the
# Again_0420/data/assets tree that ships in this workspace, which is what the
# InspireRH URDF path resolves to.
_candidates = [
    Path(__file__).parent.parent.absolute() / "assets",
    Path(__file__).parent.parent.parent.absolute() / "assets",      # ManipTrans_Lab/assets/
    Path(__file__).parent.parent.parent.parent.absolute() / "data" / "assets",
    Path(os.environ.get("MANIPTRANS_ASSET_ROOT", "")),
]

for _c in _candidates:
    if _c and _c.exists():
        ASSET_ROOT = str(_c)
        break
else:
    # Not fatal: we only warn because envs that don't reference ASSET_ROOT
    # (imitator w/o object asset, manip_sh/bih with absolute-path YAML assets)
    # work fine without it.
    import warnings
    ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")
    warnings.warn(
        f"[asset_root] no assets/ dir found at any of:\n"
        + "\n".join(f"  - {c}" for c in _candidates if str(c))
        + f"\nDefaulting ASSET_ROOT={ASSET_ROOT}. Set MANIPTRANS_ASSET_ROOT "
          "if a task needs this to resolve."
    )
