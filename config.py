"""VerdictAI configuration settings.

Centralised configuration for database paths, confidence thresholds,
entity matching, and CPM calibration parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfidenceThresholds:
    """Confidence routing thresholds for the evaluation engine."""

    # Standard thresholds
    auto_commit: float = 0.85
    mandatory_floor: float = 0.50

    # Conservative thresholds (applied when CPM data < calibration_threshold)
    conservative_auto: float = 0.90
    conservative_floor: float = 0.60


@dataclass
class Settings:
    """Application-wide settings."""

    # Database
    db_path: str = "verdict_ai.db"

    # Confidence routing
    confidence: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)

    # Entity matching
    entity_match_threshold: float = 0.85

    # CPM calibration
    cpm_calibration_threshold: int = 50


# Global settings instance
settings = Settings()
