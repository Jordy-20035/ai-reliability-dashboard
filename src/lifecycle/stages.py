"""Deployment stages for model versions."""

from __future__ import annotations

from enum import Enum


class DeploymentStage(str, Enum):
    """Typical path: development → staging → production; archived when superseded."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


# Allowed manual promotions (from_stage -> allowed to_stages)
STAGE_TRANSITIONS: dict[DeploymentStage, tuple[DeploymentStage, ...]] = {
    DeploymentStage.DEVELOPMENT: (DeploymentStage.STAGING, DeploymentStage.PRODUCTION),
    DeploymentStage.STAGING: (DeploymentStage.PRODUCTION, DeploymentStage.ARCHIVED),
    DeploymentStage.PRODUCTION: (DeploymentStage.ARCHIVED,),
    DeploymentStage.ARCHIVED: (),
}
