from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Scenario(BaseModel):
    scenario_id: str
    category: str
    prompt: str
    source: str


class GenerationRecord(BaseModel):
    run_id: str
    timestamp: datetime
    model: str
    condition: str
    scenario_id: str
    category: str
    prompt: str
    system_prompt_version: str
    decoding_params: dict[str, Any]
    latency_s: float
    output_text: str


class HeuristicFeatures(BaseModel):
    agreement_markers: int = 0
    endorsement_markers: int = 0
    refusal_markers: int = 0
    actionable_pattern: int = 0
    icant_but_pattern: int = 0


class JudgeScore(BaseModel):
    S: int = Field(ge=0, le=1)
    H: int = Field(ge=0, le=1)
    severity: int = Field(ge=0, le=3)
    rationale: str


class ScoredRecord(BaseModel):
    generation: GenerationRecord
    features: HeuristicFeatures
    judge: JudgeScore
    S2H: int


class MultiTurnRecord(BaseModel):
    run_id: str
    model: str
    condition: str
    scenario_id: str
    turn: int
    persuader_message: str
    assistant_output: str
    S: int
    H: int
    S2H: int
    severity: int
    flip_turn: int | None = None
