"""Command-line interface.

Examples
--------
::

    wavepacket list
    wavepacket run double-slit -o double-slit.gif
    wavepacket run stadium --style density --frames 150 --size 400
    wavepacket snapshot crystal --time 3.0 -o crystal.png
"""

from __future__ import annotations

import argparse
import sys
import time

from . import render, scenes


def _run(args) -> int:
    sc = scenes.get(args.scene, n=args.n) if args.n else scenes.get(args.scene)
    solver = sc.solver()
    ref = render.SmoothedReference()
    frames = [render.frame(sc.psi0, sc.V, args.style, ref(sc.psi0), sc.vmax)]
    t0 = time.perf_counter()
    for psi in solver.evolve(sc.psi0, sc.t_final, args.frames):
        frames.append(render.frame(psi, sc.V, args.style, ref(psi), sc.vmax))
        if not args.quiet:
            print(f"\r{sc.title}: t = {solver.t:6.3f} / {sc.t_final:.3f}", end="", file=sys.stderr)
    if not args.quiet:
        print(f"\n{len(frames)} frames in {time.perf_counter() - t0:.1f} s, "
              f"final norm {solver.norm(psi):.4f}", file=sys.stderr)
    out = args.output or f"{sc.name}.gif"
    render.save_gif(frames, out, fps=args.fps, size=args.size)
    print(out)
    return 0


def _snapshot(args) -> int:
    sc = scenes.get(args.scene)
    solver = sc.solver()
    psi = solver.step(sc.psi0, int(round(args.time / sc.dt)))
    img = render.to_image(render.frame(psi, sc.V, args.style, None, sc.vmax), args.size)
    out = args.output or f"{sc.name}.png"
    img.save(out)
    print(out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wavepacket", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list the built-in scenes")

    r = sub.add_parser("run", help="simulate a scene and write an animated GIF")
    r.add_argument("scene", choices=sorted(scenes.SCENES))
    r.add_argument("-o", "--output")
    r.add_argument("--frames", type=int, default=120)
    r.add_argument("--fps", type=int, default=25)
    r.add_argument("--size", type=int, default=None, help="output width in pixels")
    r.add_argument("--n", type=int, default=None, help="grid points per side")
    r.add_argument("--style", choices=["phase", "density"], default="phase")
    r.add_argument("-q", "--quiet", action="store_true")

    s = sub.add_parser("snapshot", help="write a single PNG frame at a given time")
    s.add_argument("scene", choices=sorted(scenes.SCENES))
    s.add_argument("--time", type=float, default=2.0)
    s.add_argument("-o", "--output")
    s.add_argument("--size", type=int, default=None)
    s.add_argument("--style", choices=["phase", "density"], default="phase")

    args = p.parse_args(argv)
    if args.cmd == "list":
        for name, fn in scenes.SCENES.items():
            print(f"{name:12s} {fn.__name__.replace('_', ' ')}")
        return 0
    return {"run": _run, "snapshot": _snapshot}[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
