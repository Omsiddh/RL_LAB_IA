"""Traditional scheduling policies using the existing CPU simulator."""

from .fcfs import run_fcfs
from .round_robin import run_round_robin
from .srtf import run_srtf

__all__ = ["run_fcfs", "run_srtf", "run_round_robin"]
