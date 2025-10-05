"""Environment verification script for RL setup.

Runs a comprehensive audit of the active Python interpreter, core
dependencies (PyTorch), and GPU / CUDA availability. Outputs a
human-readable report that can be redirected to documentation.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime


def line(title: str, value: str | bool | float | int | None) -> str:
    """Format a key/value pair for display."""

    return f"{title:<32}: {value}"


def check_python() -> list[str]:
    """Collect interpreter information."""

    info = ["PYTHON ENVIRONMENT"]
    info.append("-" * 80)
    info.append(line("Interpreter", sys.executable))
    info.append(line("Version", platform.python_version()))
    info.append(line("Implementation", platform.python_implementation()))
    info.append(line("Platform", platform.platform()))
    info.append("")
    return info


def check_pytorch() -> list[str]:
    """Collect PyTorch and CUDA stats."""

    info = ["PYTORCH / CUDA"]
    info.append("-" * 80)

    try:
        import torch  # type: ignore

        info.append(line("PyTorch version", torch.__version__))
        info.append(line("CUDA available", torch.cuda.is_available()))
        info.append(line("CUDA version", torch.version.cuda))
        info.append(line("GPU count", torch.cuda.device_count()))
        if torch.cuda.is_available():
            info.append(line("GPU name", torch.cuda.get_device_name(0)))
            properties = torch.cuda.get_device_properties(0)
            info.append(
                line(
                    "GPU memory (GB)",
                    round(properties.total_memory / (1024 ** 3), 2),
                )
            )
            info.append(line("Compute capability", properties.major))
        else:
            info.append("No CUDA-capable GPU detected.")
    except Exception as exc:  # pragma: no cover - defensive
        info.append(f"ERROR importing torch: {exc}")

    info.append("")
    return info


def main() -> None:
    sections = [
        f"RL ENVIRONMENT VERIFICATION REPORT - {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC",
        "=" * 80,
        "",
    ]

    sections.extend(check_python())
    sections.extend(check_pytorch())

    # Environment variables of interest (optional)
    sections.append("ENVIRONMENT VARIABLES")
    sections.append("-" * 80)
    for key in ("CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES"):
        sections.append(line(key, os.environ.get(key) or "<unset>"))

    sections.append("")
    sections.append("SUMMARY")
    sections.append("-" * 80)

    summary_checks = []
    version_tuple = tuple(map(int, platform.python_version_tuple()))
    summary_checks.append(version_tuple >= (3, 10))

    try:
        import torch  # type: ignore

        torch_ok = torch.cuda.is_available() and torch.version.cuda is not None
        torch_version_ok = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 0)
        cuda_version = torch.version.cuda or "0"
        cuda_version_tuple = tuple(map(int, cuda_version.split(".")))
        cuda_ok = cuda_version_tuple >= (12, 1)
        gpu_ok = torch.cuda.device_count() >= 1

        summary_checks.extend([torch_version_ok, cuda_ok, gpu_ok, torch_ok])

        sections.append(line("Python >= 3.10", summary_checks[0]))
        sections.append(line("PyTorch >= 2.0", torch_version_ok))
        sections.append(line("CUDA >= 12.1", cuda_ok))
        sections.append(line("GPU detected", gpu_ok))
        sections.append(line("CUDA available", torch.cuda.is_available()))
    except Exception as exc:  # pragma: no cover
        sections.append(f"Torch verification failed: {exc}")

    sections.append("")
    sections.append("All checks passed" if all(summary_checks) else "Some checks failed")

    print("\n".join(sections))


if __name__ == "__main__":
    main()