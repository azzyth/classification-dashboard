"""Pydantic schemas for the classification dashboard API."""
from typing import Optional, List, Dict
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: Dict[str, float]
    true_label: Optional[str] = None


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: List[float]
    class_names: List[str]
    confidence: float


class DataRecord(BaseModel):
    id: int
    features: Dict[str, float]
    true_label: Optional[str]
    predicted_label: str
    probabilities: List[float]
    confidence: float
    created_at: str


class DataInsertResponse(BaseModel):
    record: DataRecord
    metrics_updated: bool


class MetricsResponse(BaseModel):
    accuracy: float
    macro_f1: float
    micro_f1: float
    per_class: Dict[str, Dict[str, float]]
    n_samples: int
    n_labeled: int
    model_version: int
    trained_at: Optional[str]


class ConfusionMatrixResponse(BaseModel):
    matrix: List[List[int]]
    class_names: List[str]
    per_class_f1: Dict[str, float]


class DistributionResponse(BaseModel):
    class_names: List[str]
    prior: List[float]
    predicted: List[float]


class VisualizationItem(BaseModel):
    id: int
    viz_type: str
    feature_name: str
    title: str
    image_base64: str

class CSVUploadResponse(BaseModel):
    upload_id: int
    filename: str
    n_rows: int
    n_cols: int
    feature_names: List[str]
    target_name: str
    feature_types: Dict[str, str]
    class_distribution: Dict[str, int]
    visualizations: List[VisualizationItem]
    model_version: int
    metrics: Optional[Dict] = None

class CSVAnalysisResponse(BaseModel):
    n_rows: int
    n_cols: int
    feature_names: List[str]
    target_name: str
    feature_types: Dict[str, str]
    class_distribution: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    record_count: int
    labeled_count: int
    model_version: int
    class_names: List[str]
