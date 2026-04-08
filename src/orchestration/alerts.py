"""Operational alerting for orchestration runs (webhook notifications)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from urllib import error, request

from .actions import PipelineContext

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_alert_payload(ctx: PipelineContext) -> dict[str, Any]:
    """Build a compact alert payload for downstream channels."""
    return {
        "event": "drift_policy_triggered",
        "sent_at": _utc_now(),
        "scenario": str(ctx.metadata.get("scenario", "")),
        "trigger_reasons": list(ctx.trigger_reasons),
        "summary": dict(ctx.report.summary),
        "metadata": dict(ctx.metadata),
    }


def post_json_webhook(url: str, payload: dict[str, Any], *, timeout_seconds: int = 5) -> None:
    """POST JSON payload to webhook URL (best-effort)."""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as resp:  # noqa: S310 - caller-controlled URL by config
        status = getattr(resp, "status", 200)
        if status >= 400:
            raise RuntimeError(f"Webhook responded with status={status}")


class WebhookAlertAction:
    """Send a webhook alert when drift policy is triggered."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url.strip()

    def run(self, ctx: PipelineContext) -> None:
        if not ctx.policy_triggered:
            return
        payload = build_alert_payload(ctx)
        try:
            post_json_webhook(self.webhook_url, payload)
            logger.warning("Webhook alert sent for scenario=%s", payload["scenario"])
        except (error.URLError, TimeoutError, RuntimeError) as exc:
            logger.exception("Webhook alert failed: %s", exc)
