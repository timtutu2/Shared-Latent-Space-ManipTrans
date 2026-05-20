#!/usr/bin/env python3
"""Create a simple fixed table USD for ManipTrans stage-2 envs.

The envs spawn table at z=0.4, and TABLE_SURFACE_Z is 0.415.
So this table is built with local top surface at +0.015 m.
"""

from __future__ import annotations

import argparse
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def build_table_usd(out_path: str, length: float, width: float, thickness: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    table = UsdGeom.Xform.Define(stage, "/World/table")

    top = UsdGeom.Cube.Define(stage, "/World/table/top")
    top.CreateSizeAttr(1.0)
    top.AddScaleOp().Set(Gf.Vec3f(length, width, thickness))
    # Keep local top surface at +0.015m when thickness=0.03.
    top.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.015 - thickness * 0.5))
    top.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.30, 0.25)])

    # Physics APIs: table is fixed by env fix_root=True + kinematic rigid props.
    table_prim = table.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(table_prim)
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())

    stage.Save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="/home/david/david/Again_0420/data/assets/furniture/table.usd",
        help="Output table USD path.",
    )
    parser.add_argument("--length", type=float, default=0.8, help="Table top length (x).")
    parser.add_argument("--width", type=float, default=0.6, help="Table top width (y).")
    parser.add_argument("--thickness", type=float, default=0.03, help="Table top thickness (z).")
    args = parser.parse_args()

    if args.thickness <= 0.0:
        raise ValueError("--thickness must be > 0")

    build_table_usd(
        out_path=args.out,
        length=args.length,
        width=args.width,
        thickness=args.thickness,
    )
    print(f"[create_table_usd] wrote: {args.out}")


if __name__ == "__main__":
    main()
