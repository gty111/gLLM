"""Small runtime helpers shared by FlashInfer-backed operators."""

import os
import shutil


def ensure_ninja_on_path() -> None:
    """Expose FlashInfer's bundled Ninja when Python was launched by path.

    Calling ``.../env/bin/python`` does not activate the environment, so its
    scripts directory may be absent from ``PATH`` even though the ``ninja``
    Python package is installed. FlashInfer's JIT invokes the executable via a
    subprocess and therefore needs that directory on ``PATH``.
    """
    if shutil.which("ninja") is not None:
        return
    try:
        import ninja
    except ImportError:
        return

    ninja_executable = os.path.join(ninja.BIN_DIR, "ninja")
    if os.path.isfile(ninja_executable):
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = (
            ninja.BIN_DIR
            if not current_path
            else ninja.BIN_DIR + os.pathsep + current_path
        )

