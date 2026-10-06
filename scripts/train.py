"""Train `trainer.learner` on `task.name`.

Thin launcher: the real entry point is :func:`robonuke_rl_core.train.main`, which takes a
``setup`` hook a project uses to register its tasks and config sections. See the
robonuke_project_template repo for the project-side version of this script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robonuke_rl_core.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
