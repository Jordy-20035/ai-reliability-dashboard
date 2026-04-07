"""Schedule recurring drift checks (APScheduler)."""

from __future__ import annotations

import logging
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)


def start_interval_scheduler(
    job: Callable[[], None],
    *,
    interval_seconds: int,
) -> BackgroundScheduler:
    """Run `job` every `interval_seconds` in a background thread."""
    sched = BackgroundScheduler()
    sched.add_job(job, "interval", seconds=interval_seconds, id="drift_check", replace_existing=True)
    sched.start()
    logger.info("Scheduler started: interval=%ss", interval_seconds)
    return sched
