"""Write slides/numbers.tex and slides/fonts.tex, check the figures, compile slides/lab2_presentation.pdf.

    python scripts/build_slides.py [--tectonic PATH]

Every number on the slides is a macro from numbers.tex, computed here from
results/; the claims the slide text makes about them (which elements are kept,
how much narrower the naive Fisher width is) are asserted before writing, so a
rerun with different results fails instead of compiling stale prose. Figures
come from the notebook run (lab02/report.py); the build stops if one the slides
include is missing or older than the results. Fonts are matplotlib's DejaVu, so
no system fonts are needed; tectonic is looked up in .tools/tectonic/bin first.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab02 import config, graph, report  # noqa: E402
from lab02.divergence import THRESHOLD  # noqa: E402
from lab02.experiment import NAMES, critical_fraction  # noqa: E402
from lab02.intervention import attenuation  # noqa: E402
from lab02.plots import EDGES, VERTICES, _count, _fmt  # noqa: E402

SLIDES = ROOT / "slides"
TEX = SLIDES / "lab2_presentation.tex"


def _math(text: str) -> str:
    return rf"\ensuremath{{{text}}}"


def _sci(x: float, digits: int = 2) -> str:
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 2:
        return f"{x:.{digits}g}"
    return rf"{x / 10**exp:.1f}\cdot10^{{{exp}}}"


def _pm(mean: float, std: float) -> str:
    return _fmt(mean, std).replace("±", r"\pm")


def numbers(R: report.Results) -> dict[str, str]:
    rep = R.reports
    mc = R.monte_carlo["summary"]
    count = len(R.monte_carlo["realizations"])

    def est(n, e, key="l2/godambe"):
        b = rep[n]["base"][key]
        i = NAMES.index(e)
        return b["mean"][i], float(np.sqrt(b["cov"][i][i]))

    def test(n, e, step=0):
        return rep[n]["steps"][step]["tests"][e]

    # Claims the slide text makes in words.
    assert rep["syn_a"]["keep"] == {"A": True, "V": True, "D": False}
    assert rep["syn_b"]["keep"] == {"A": True, "V": True, "D": True}
    assert rep["real"]["keep"] == {"A": True, "V": False, "D": False}
    assert all(s["keep"] == rep["real"]["keep"] for s in rep["real"]["scales"])
    assert all(t["unanimous"] for step in rep["real"]["steps"] for t in step["tests"].values())
    for e in ("V", "D"):  # "Q narrower, centre moved by less than one sigma"
        t = test("real", e)
        assert t["sigma_after"] < t["sigma"] and abs(t["theta_after"] - t["theta"]) < t["sigma"]
        hm, hs = est("real", e, "huber/godambe")  # "Huber keeps V and D within one sigma of zero"
        assert abs(hm) < hs
    assert (
        rep["syn_a"]["critical_q"]["A"]["q"]["upper_bound"]
        and rep["syn_a"]["critical_q"]["V"]["q"]["upper_bound"]
    )
    # "V and D: the shift grows with the intervention but stays below the threshold,
    # since the centre can move at most to zero and zero is within one sigma"
    plateau = {}
    for e in ("V", "D"):
        curves = rep["real"]["curves"][e]
        plateau[e] = max(max(curves[s]["median"]) for s in ("q", "k"))
        assert curves["q"]["median"][-1] > curves["q"]["median"][0] and plateau[e] < THRESHOLD
        mean, sigma = est("real", e)
        assert abs(mean) < sigma

    rho = attenuation(10.0, 1.0)
    z_eff = float(np.sqrt(2 * THRESHOLD) / rho)
    real_q = rep["real"]["critical_q"]["A"]
    placebo = critical_fraction(rep["real"]["curves"]["A"]["q_placebo"])["value"]
    narrow = {
        n: [est(n, e)[1] / est(n, e, "l2/fisher")[1] for e in NAMES] for n in ("syn_a", "real")
    }
    under_fisher = [1 / r for r in mc["l2"]["ratio"]["fisher"]]
    expected = {m: norm.sf(z_eff * mc["l2"]["ratio"][m][2]) * count for m in ("fisher", "godambe")}
    sig_grid = np.array(rep["real"]["leaf_variant"]["sigma_conditional"]) / np.array(
        rep["real"]["leaf_variant"]["sigma_marginal"]
    )
    leaf_rows = rep["real"]["leaf_variant"]["rows"]
    leaf_kl = float(
        np.median([r["shift"] for r in leaf_rows if r["leaf"] == "n01" and r["q"] == 0.05])
    )
    curve_a = rep["real"]["curves"]["A"]["q"]
    window_kl = curve_a["median"][curve_a["x"].index(0.05)]
    losses = []
    for n, leaves in (("syn_a", ("n01", "n03", "n13")), ("real", ("n01", "n03"))):
        rows = rep[n]["leaf_variant"]["rows"]
        S = rep[n]["removal_kl"]["l2/godambe"]
        for leaf in leaves:
            element = "V" if leaf == "n13" else "A"
            c = rep[n]["curves"][element]["q"]
            for q in (0.05, 0.5):
                grid_rho = abs(
                    np.median([r["rho"] for r in rows if r["leaf"] == leaf and r["q"] == q])
                )
                losses.append(np.sqrt(c["median"][c["x"].index(q)] / S[element]) / grid_rho)
    diffs = {
        n: graph.graph_diff(graph.candidate_tree(), graph.decided_tree(rep[n]["keep"])) for n in rep
    }
    steps = rep["real"]["steps"]
    theta_a = [s["theta"]["A"] for s in rep["real"]["scales"]]

    v = {
        "Threshold": f"{THRESHOLD:.2f}",
        "Zeff": f"{z_eff:.2f}",
        "MCcount": str(count),
        "MCFisherD": f"{round(mc['keep_share']['l2/fisher']['D'] * count)}/{count}",
        "MCGodambeD": f"{round(mc['keep_share']['l2/godambe']['D'] * count)}/{count}",
        "MCFisherDExpected": f"{expected['fisher']:.1f}",
        "MCGodambeDExpected": f"{expected['godambe']:.1f}",
        "MCFisherUnderMin": f"{min(under_fisher):.0f}",
        "MCFisherUnderMax": f"{max(under_fisher):.0f}",
        "MCGodambeUnderA": f"{1 / mc['l2']['ratio']['godambe'][0]:.1f}",
        "NarrowSynMin": f"{min(narrow['syn_a']):.1f}",
        "NarrowSynMax": f"{max(narrow['syn_a']):.1f}",
        "NarrowRealMin": f"{min(narrow['real']):.1f}",
        "NarrowRealMax": f"{max(narrow['real']):.1f}",
        "SynAKLD": _math(_sci(test("syn_a", "D")["kl_shift"])),
        "SynBThetaD": _math(_pm(*est("syn_b", "D"))),
        "SynBKLD": _math(_sci(test("syn_b", "D")["kl_shift"], 3)),
        "SynBQstarD": rf"{rep['syn_b']['critical_q']['D']['q']['value'] * 100:.1f}\,\%",
        "RealWindows": str(rep["real"]["meta"]["windows"]),
        "RealQstar": _math(_sci(real_q["q"]["value"])),
        "RealQstarWindows": str(
            int(np.ceil(real_q["q"]["value"] * rep["real"]["meta"]["windows"]))
        ),
        "HuberRatio": f"{real_q['q_huber']['value'] / real_q['q']['value']:.0f}",
        "PlaceboRatio": f"{placebo / real_q['q']['value']:.0f}",
        "HuberShiftA": f"{abs(est('real', 'A', 'huber/godambe')[0] - est('real', 'A')[0]) / est('real', 'A')[1]:.1f}",
        "VinSigma": f"{1 / est('real', 'V')[1]:.2f}",
        "OrdersAV": f"{np.log10(test('real', 'A')['kl_shift'] / test('real', 'V')['kl_shift']):.1f}",
        "RealPlateauV": f"{plateau['V']:.2f}",
        "RealPlateauD": f"{plateau['D']:.2f}",
        "GridSigmaReal": f"{sig_grid.mean():.2f}",
        "LeafLossMin": f"{min(losses):.0f}",
        "LeafLossMax": f"{float(f'{max(losses):.1g}'):.0f}",  # one significant digit: a range, not a measurement
        "LeafKL": f"{leaf_kl:.2f}",
        "WindowKL": _math(_sci(window_kl)),
        "ThetaAmin": f"{min(theta_a):.2f}",
        "ThetaAmax": f"{max(theta_a):.2f}",
        "StepKLV": f"{steps[1]['tests']['V']['kl_shift']:.2f}",
        "RsqFull": f"{steps[0]['r2']:.4f}",
        "RsqFinal": f"{steps[-1]['r2']:.4f}",
        "GraphBeforeText": f"{_count(diffs['real'].before[0], VERTICES)}, {_count(diffs['real'].before[1], EDGES)}",
        "GraphAfterRealText": f"{_count(diffs['real'].after[0], VERTICES)}, {_count(diffs['real'].after[1], EDGES)}",
        "NTests": str(_test_count()),
    }
    for e in NAMES:
        t = test("real", e)
        v[f"RealTheta{e}"] = _math(_pm(t["theta"], t["sigma"]))
        v[f"RealQ{e}"] = _math(_pm(t["theta_after"], t["sigma_after"]))
        v[f"RealKL{e}"] = _math(_sci(t["kl_shift"], 2))
        v[f"RealShape{e}"] = _math(f"{t['kl_shape']:.2f}")
    verdict = {True: r"\textcolor{keep}{оставить}", False: r"\textcolor{drop}{удалить}"}
    rows = []
    for n in rep:
        cells = [
            report.SHORT[n],
            *(verdict[rep[n]["keep"][e]] for e in NAMES),
            "{} / {}".format(*diffs[n].after),
        ]
        rows.append("    " + " & ".join(cells) + r" \\")
    v["DecisionRows"] = "\n".join(rows)
    return v


def _test_count() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = re.search(r"(\d+) tests? collected", out)
    if not found:
        raise RuntimeError(f"pytest did not report a test count:\n{out[-2000:]}")
    return int(found.group(1))


def write_numbers(values: dict[str, str]) -> None:
    lines = ["% Written by scripts/build_slides.py from results/; do not edit."]
    for name, value in values.items():
        if not name.isalpha():
            raise ValueError(f"TeX macro names are letters only: {name!r}")
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    (SLIDES / "numbers.tex").write_text("\n".join(lines) + "\n")


def write_fonts() -> None:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    path = f"Path={ttf}/, Extension=.ttf"
    sans = f"{path}, BoldFont=DejaVuSans-Bold, ItalicFont=DejaVuSans-Oblique, BoldItalicFont=DejaVuSans-BoldOblique"
    serif = f"{path}, BoldFont=DejaVuSerif-Bold, ItalicFont=DejaVuSerif-Italic, BoldItalicFont=DejaVuSerif-BoldItalic"
    mono = f"{path}, BoldFont=DejaVuSansMono-Bold, ItalicFont=DejaVuSansMono-Oblique"
    (SLIDES / "fonts.tex").write_text(
        "% Written by scripts/build_slides.py: the template's DejaVu fonts, taken from matplotlib.\n"
        rf"\setsansfont{{DejaVuSans}}[{sans}, Scale=0.88]"
        "\n"
        rf"\setmainfont{{DejaVuSerif}}[{serif}, Scale=0.88]"
        "\n"
        rf"\setmonofont{{DejaVuSansMono}}[{mono}, Scale=0.8]"
        "\n"
        rf"\newfontfamily\cyrillicfontsf{{DejaVuSans}}[{sans}, Scale=0.88]"
        "\n"
        rf"\newfontfamily\cyrillicfont{{DejaVuSerif}}[{serif}, Scale=0.88]"
        "\n"
        rf"\newfontfamily\cyrillicfonttt{{DejaVuSansMono}}[{mono}, Scale=0.8]"
        "\n"
    )


def check_figures() -> list[str]:
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", TEX.read_text())
    newest_result = max((config.RESULTS / name).stat().st_mtime for name in report.RESULT_FILES)
    problems = []
    for name in names:
        path = config.FIGURES / f"{name}.pdf"
        if name not in report.FIGURES:
            problems.append(f"{name}: not a figure of lab02/report.py")
        elif not path.exists():
            problems.append(f"{name}: missing")
        elif path.stat().st_mtime < newest_result:
            problems.append(f"{name}: older than the results")
    if problems:
        raise SystemExit(
            "Figures are not ready (run scripts/run_notebook.py):\n  " + "\n  ".join(problems)
        )
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", default=None)
    args = parser.parse_args()
    tectonic = (
        args.tectonic
        or shutil.which("tectonic", path=str(ROOT / ".tools" / "tectonic" / "bin"))
        or shutil.which("tectonic")
    )
    if tectonic is None:
        raise SystemExit(
            "tectonic not found: conda create -p .tools/tectonic -c conda-forge tectonic"
        )
    names = check_figures()
    write_numbers(numbers(report.load()))
    write_fonts()
    subprocess.run([tectonic, "-X", "compile", TEX.name], cwd=SLIDES, check=True)
    print(f"wrote {TEX.with_suffix('.pdf').relative_to(ROOT)} with {len(names)} figures")


if __name__ == "__main__":
    main()
