import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def pytest_sessionfinish(session, exitstatus):
    """With ``BC_FAIL_ON_SKIP=1``, a skipped test fails the run.

    CI sets this. Without it a job can go green having executed nothing: the
    optional-backend suites skip themselves when their toolchain is missing,
    which is right locally and useless as a gate. The core suite must have zero
    skips by construction, and the rust/jax jobs install the toolchain first, so
    a skip in CI means the thing under test never ran.
    """
    if os.environ.get("BC_FAIL_ON_SKIP") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = len(reporter.stats.get("skipped", []))
    if skipped:
        reporter.write_line(
            f"BC_FAIL_ON_SKIP=1: {skipped} skipped test(s) -> failing the run",
            red=True)
        session.exitstatus = 1
