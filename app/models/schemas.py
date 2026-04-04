from typing import List, Optional
from pydantic import BaseModel


class PredictRequest(BaseModel):
    eeg_signal: List[float]   # raw EEG samples at 100 Hz
    smooth: bool = True


class EpochResult(BaseModel):
    epoch: int
    stage_id: int
    stage_name: str
    confidence: float


class Metrics(BaseModel):
    recording_minutes: float
    tst_min: float
    sleep_latency_min: float
    sleep_efficiency: float
    waso_min: float
    pct_n3: float
    pct_rem: float
    fragmentation_index: float
    confidence_mean: float
    sleep_quality_score: float


class PredictResponse(BaseModel):
    epochs: List[EpochResult]
    metrics: Metrics


class ExplainRequest(BaseModel):
    metrics: Metrics
    language: str = "english"   # "english" or "hindi"


class ExplainResponse(BaseModel):
    explanation: str
    tips: List[str]
    ai_available: bool


class QuickScoreResponse(BaseModel):
    predicted_score: float
    score_range: str = "0-100"
    model: str = "RandomForest (spectral EEG features)"
    note: str = "Fast estimate. Use /predict/edf for full staging + accurate score."


class JobResponse(BaseModel):
    job_id: str
    status: str   # "processing"


class JobStatusResponse(BaseModel):
    status: str                          # "processing" | "done" | "failed"
    result: Optional[PredictResponse] = None
    error: Optional[str] = None