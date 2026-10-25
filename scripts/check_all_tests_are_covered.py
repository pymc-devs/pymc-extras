"""
Check that every test file is run by the CI matrix in .github/workflows/test.yml.

The matrix splits the suite into shards given as pytest arguments: directories, files, node ids,
and ``--ignore`` paths. Every test file must be collected by at least one shard on some platform,
and by at most one shard per platform and linker. Run it with ``pre-commit run
check-no-tests-are-ignored --all-files``.
"""

import itertools
import sys

from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "test.yml"
JOBS_WITHOUT_TESTS = {"pre-commit", "all_tests"}


def test_files() -> set[str]:
    return {path.relative_to(REPO).as_posix() for path in (REPO / "tests").glob("**/test_*.py")}


def expand(path: str, all_tests: set[str]) -> set[str]:
    """Return the test files a pytest path argument collects."""
    file_path = path.split("::")[0]
    if file_path.endswith(".py"):
        return {file_path}
    return {test for test in all_tests if test.startswith(file_path.rstrip("/") + "/")}


def files_run_by(shard: str, all_tests: set[str]) -> set[str]:
    tokens = shard.split()
    ignored: set[str] = set()
    included: set[str] = set()
    for token in tokens:
        if token.startswith("--ignore="):
            ignored |= expand(token.removeprefix("--ignore="), all_tests)
        elif token.startswith("--ignore"):
            continue
        elif tokens[max(tokens.index(token) - 1, 0)] == "--ignore":
            ignored |= expand(token, all_tests)
        else:
            included |= expand(token, all_tests)
    if not included:
        included = set(all_tests)
    return included - ignored


def main() -> int:
    all_tests = test_files()
    runs: Counter[tuple[str, str, str]] = Counter()
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    for job_name, job in jobs.items():
        if job_name in JOBS_WITHOUT_TESTS:
            continue
        matrix = job["strategy"]["matrix"]
        for os_name, linker, shard in itertools.product(
            matrix["os"], matrix["linker"], matrix["test-subset"]
        ):
            for test in files_run_by(shard, all_tests):
                runs[(test, os_name, linker)] += 1

    never_run = sorted(test for test in all_tests if not any(key[0] == test for key in runs))
    run_twice = sorted(
        f"{test} ({os_name}, {linker})" for (test, os_name, linker), n in runs.items() if n > 1
    )

    if never_run:
        print(
            f"{len(never_run)} test files are not run by any CI shard:\n  " + "\n  ".join(never_run)
        )
    if run_twice:
        print(
            f"{len(run_twice)} test files run twice under the same OS and linker:\n  "
            + "\n  ".join(run_twice)
        )
    if never_run or run_twice:
        return 1
    print(f"All {len(all_tests)} test files run exactly once per OS and linker where scheduled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
