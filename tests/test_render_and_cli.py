"""Smoke tests for rendering, scenes and the command line."""

import numpy as np
import pytest
from PIL import Image

import wavepacket as wp
from wavepacket import cli, render, scenes


def test_domain_colour_encodes_phase_as_hue():
    psi = np.array([[1.0, 1j, -1.0, -1j]])
    rgb = render.domain_color(psi, ref=1.0, gamma=1.0, saturation=1.0)
    assert rgb.shape == (1, 4, 3)
    np.testing.assert_allclose(rgb[0, 0], [1, 0, 0], atol=1e-12)  # phase 0 -> red
    assert rgb[0, 2, 0] < 0.01 and rgb[0, 2, 1] > 0.99  # phase pi -> cyan
    assert np.all(render.domain_color(np.zeros((2, 2)), ref=1.0) == 0)


def test_frame_is_uint8_and_marks_walls():
    g = wp.Grid.square(64, 10.0)
    V = wp.potentials.slits(g, 2, 2.0, 0.5, height=100.0)
    img = render.frame(wp.gaussian_packet(g, (-2, 0), (3, 0), 1.0), V)
    assert img.dtype == np.uint8 and img.shape == (64, 64, 3)
    wall_col = np.argmin(np.abs(g.axes[0]))
    assert img[:, wall_col].mean() > 20  # outline/fill visible


@pytest.mark.parametrize("name", sorted(scenes.SCENES))
def test_every_scene_builds_and_conserves_norm_briefly(name):
    sc = scenes.get(name, n=128)
    s = sc.solver()
    psi = s.step(sc.psi0, 5)
    assert sc.psi0.shape == sc.grid.shape
    assert s.norm(psi) == pytest.approx(s.norm(sc.psi0), rel=1e-3)


def test_unknown_scene_raises():
    with pytest.raises(KeyError):
        scenes.get("warp-drive")


def test_cli_run_writes_a_gif(tmp_path, capsys):
    out = tmp_path / "osc.gif"
    assert cli.main(["run", "oscillator", "--n", "64", "--frames", "4", "-q", "-o", str(out)]) == 0
    im = Image.open(out)
    assert im.n_frames == 5


def test_cli_snapshot_and_list(tmp_path, capsys):
    out = tmp_path / "tun.png"
    assert cli.main(["snapshot", "oscillator", "--time", "0.1", "-o", str(out)]) == 0
    assert Image.open(out).size == (256, 256)
    assert cli.main(["list"]) == 0
    assert "double-slit" in capsys.readouterr().out
