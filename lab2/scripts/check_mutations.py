"""Mutation check of the tests: every listed defect must make the suite fail.

    python scripts/check_mutations.py

Each mutation is applied to a temporary copy of the package and the tests, never
to the working tree, so an interrupted run leaves the sources intact. The
unmutated copy runs first: a red baseline would make every mutation look caught.
Exit status 1 if the baseline fails or any mutation survives.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPIED = ("lab02", "tests", "pytest.ini")
LINKED = ("third_party",)  # read-only for the tests: linked, not copied

# (file, original text, mutant text, defect the mutant models)
MUTATIONS = (
    (
        "lab02/intervention.py",
        "return 1.0 - (1 - q + q * k * SQRT_2_OVER_PI) / (1 - q + q * k * k)",
        "return 1.0 - (1 - q + q * k) / (1 - q + q * k * k)",
        "attenuation without E|r|",
    ),
    (
        "lab02/intervention.py",
        "side = np.where(column >= cond.mean, 1.0, -1.0)",
        "side = 1.0",
        "values always moved upwards",
    ),
    (
        "lab02/intervention.py",
        "moved = cond.mean + side * k * cond.sigma[cond.component]",
        "moved = cond.mean + side * k * cond.sigma[0]",
        "sigma of component 0 for every component",
    ),
    (
        "lab02/intervention.py",
        "mask[rng.choice(n_windows, int(np.ceil(q * n_windows)), replace=False)] = True",
        "mask[rng.choice(n_windows, int(q * n_windows), replace=False)] = True",
        "floor instead of ceil for the moved windows",
    ),
    (
        "lab02/experiment.py",
        "significant = kl_shift > THRESHOLD",
        "significant = kl_shift >= THRESHOLD",
        "non-strict threshold",
    ),
    (
        "lab02/experiment.py",
        '                    **rule(element, float(P.mean[i]), part["shift"]),',
        '                    **rule(element, float(P.mean[i]), part["total"]),',
        "decision on the total KL instead of the shift",
    ),
    (
        "lab02/experiment.py",
        'removed = min(failing, key=lambda e: tests[e]["kl_shift"]) if failing else None',
        'removed = max(failing, key=lambda e: tests[e]["kl_shift"]) if failing else None',
        "strongest failing element removed first",
    ),
    (
        "lab02/experiment.py",
        'return {"value": float(xs[0]), "upper_bound": True}',
        'return {"value": float(xs[0]), "upper_bound": False}',
        "grid edge passed off as the critical fraction",
    ),
    (
        "lab02/graph.py",
        'ADMISSIBLE_SIGN = {"A": 1.0, "V": 1.0, "D": 1.0}',
        'ADMISSIBLE_SIGN = {"A": 1.0, "V": -1.0, "D": 1.0}',
        "wrong admissible sign of the viscosity",
    ),
    (
        "lab02/estimation.py",
        "    np.add.at(scores, inverse, X * fit.psi[:, None])\n"
        "    return _gaussian(fit, g / (g - 1) * bread @ (scores.T @ scores) @ bread)",
        "    scores = X * fit.psi[:, None]\n"
        "    return _gaussian(fit, bread @ (scores.T @ scores) @ bread)",
        "Godambe information without clusters",
    ),
    (
        "lab02/divergence.py",
        "    delta = np.linalg.solve(chol, p.mean - q.mean)",
        "    delta = np.linalg.solve(np.linalg.cholesky(p.cov), p.mean - q.mean)",
        "KL shift in units of the width after the intervention, not before",
    ),
    (
        "lab02/synthetic.py",
        "return self.project(nonlinear) - (self.nu * self.k2 + self.alpha) * uh",
        "return self.project(nonlinear) - (self.nu * self.k2) * uh",
        "friction ignored by the DNS",
    ),
    (
        "lab02/plots.py",
        "if n % 10 == 1 and n % 100 != 11:",
        "if n % 10 == 1:",
        "Russian plural of 11",
    ),
    (
        "lab02/plots.py",
        "elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:",
        "elif 2 <= n % 10 <= 4:",
        "Russian plural of 12-14",
    ),
)


def run_suite(mutation=None) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in COPIED:
            source = ROOT / name
            if source.is_dir():
                shutil.copytree(source, tmp / name, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(source, tmp / name)
        for name in LINKED:
            (tmp / name).symlink_to(ROOT / name, target_is_directory=True)
        if mutation:
            path, old, new, label = mutation
            target = tmp / path
            text = target.read_text()
            if text.count(old) != 1:
                raise SystemExit(
                    f"{label}: original text found {text.count(old)} times in {path}, "
                    "the mutation no longer matches the code"
                )
            target.write_text(text.replace(old, new))
        return subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-x", "--no-header", "--color=no"],
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )


def failed_test(run: subprocess.CompletedProcess) -> str:
    lines = [line for line in run.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
    return lines[0] if lines else run.stdout.strip().splitlines()[-1]


def main():
    baseline = run_suite()
    if baseline.returncode != 0:
        print(baseline.stdout[-3000:])
        raise SystemExit("baseline: the unmutated suite fails")
    print(f"baseline: {baseline.stdout.strip().splitlines()[-1]}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(run_suite, MUTATIONS))
    survived = 0
    for (_, _, _, label), run in zip(MUTATIONS, runs):
        if run.returncode == 0:
            survived += 1
            print(f"SURVIVED  {label}")
        else:
            print(f"caught    {label}: {failed_test(run)}")
    print(f"{len(MUTATIONS) - survived}/{len(MUTATIONS)} mutations caught")
    if survived:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
