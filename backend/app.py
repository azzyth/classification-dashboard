"""FastAPI application for the real-time classification dashboard."""
import os
import sys
import io
import json
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from database import (init_db, insert_record, get_all_records, get_record_count,
                      get_labeled_records, clear_all, get_model_state, update_model_state,
                      save_upload_meta, get_upload_meta, save_visualization, get_visualizations,
                      clear_uploads)
from model import get_model
from visualizations import analyze_csv, generate_visualizations
from schemas import (
    PredictRequest, PredictResponse, DataRecord, DataInsertResponse,
    MetricsResponse, ConfusionMatrixResponse, DistributionResponse, HealthResponse,
    VisualizationItem, CSVUploadResponse, CSVAnalysisResponse
)

app = FastAPI(title="Classification ML Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
def startup():
    init_db()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Serve frontend static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Classification ML Dashboard API. Frontend not found."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    pred_class, probs, confidence = model.predict(req.features)
    return PredictResponse(
        predicted_class=pred_class,
        probabilities=probs,
        class_names=model.class_names,
        confidence=round(confidence, 4),
    )


@app.post("/data", response_model=DataInsertResponse)
def insert_data(req: PredictRequest):
    import traceback
    try:
        model = get_model()
        pred_class, probs, confidence = model.predict(req.features)

        rid = insert_record(req.features, req.true_label, pred_class, probs, confidence)
        record = {
            "id": rid,
            "features": req.features,
            "true_label": req.true_label,
            "predicted_label": pred_class,
            "probabilities": probs,
            "confidence": confidence,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        state = get_model_state()
        labeled = get_labeled_records()
        metrics_updated = False
        if len(labeled) >= 4 and state["version"] == 0:
            metrics, cm, err = model.train(labeled)
            if err is None:
                update_model_state(1, datetime.now(timezone.utc).isoformat(), metrics, cm, model.class_names)
                metrics_updated = True

        return DataInsertResponse(record=DataRecord(**record), metrics_updated=metrics_updated)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data", response_model=list)
def list_data(limit: int = 100, offset: int = 0):
    records = get_all_records(limit, offset)
    return [DataRecord(**r) for r in records]


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    model = get_model()
    records = get_all_records(limit=10000)
    labeled = [r for r in records if r["true_label"] is not None]
    metrics = model.compute_metrics(records)
    state = get_model_state()

    if metrics is None:
        metrics = {
            "accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0,
            "per_class": {}
        }

    return MetricsResponse(
        accuracy=metrics.get("accuracy", 0.0),
        macro_f1=metrics.get("macro_f1", 0.0),
        micro_f1=metrics.get("micro_f1", 0.0),
        per_class=metrics.get("per_class", {}),
        n_samples=get_record_count(),
        n_labeled=len(labeled),
        model_version=state["version"],
        trained_at=state["trained_at"],
    )


@app.get("/confusion-matrix", response_model=ConfusionMatrixResponse)
def get_confusion_matrix():
    model = get_model()
    records = get_all_records(limit=10000)
    cm, per_class_f1 = model.compute_confusion_matrix(records)
    if cm is None:
        cm = [[0]]
        per_class_f1 = {}
    state = get_model_state()
    class_names = state["class_names"] or model.class_names
    return ConfusionMatrixResponse(
        matrix=cm,
        class_names=class_names,
        per_class_f1=per_class_f1 or {},
    )


@app.post("/retrain")
def retrain():
    model = get_model()
    labeled = get_labeled_records()
    if len(labeled) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 labeled records to retrain")
    metrics, cm, err = model.train(labeled)
    if err:
        raise HTTPException(status_code=400, detail=err)
    state = get_model_state()
    new_version = state["version"] + 1
    update_model_state(new_version, datetime.now(timezone.utc).isoformat(), metrics, cm, model.class_names)
    return {"status": "success", "version": new_version, "metrics": metrics}


@app.get("/distribution", response_model=DistributionResponse)
def get_distribution():
    model = get_model()
    records = get_all_records(limit=10000)
    prior, predicted = model.get_distribution(records)
    return DistributionResponse(
        class_names=model.class_names,
        prior=prior,
        predicted=predicted,
    )


@app.delete("/data")
def delete_data():
    clear_all()
    # Reset model singleton
    import model as m
    m._model_instance = None
    return {"status": "cleared"}


# --- CSV Upload & Visualization Endpoints ---

@app.post("/upload-csv", response_model=CSVUploadResponse)
async def upload_csv(file: UploadFile = File(...), target: str = None):
    """Upload a CSV, analyze, train model, generate visualizations."""
    import traceback
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

        if len(df) < 4:
            raise HTTPException(status_code=400, detail="CSV must have at least 4 rows")

        analysis = analyze_csv(df)
        target_col = target if target and target in df.columns else analysis['target_name']

        if target_col != analysis['target_name']:
            analysis['target_name'] = target_col
            analysis['feature_names'] = [c for c in df.columns if c != target_col]
            analysis['feature_types'] = {}
            for col in analysis['feature_names']:
                from visualizations import detect_feature_type
                analysis['feature_types'][col] = detect_feature_type(df[col])

        upload_id = save_upload_meta(
            file.filename, analysis['n_rows'], analysis['n_cols'],
            analysis['feature_names'], target_col,
            analysis['feature_types'], analysis['class_distribution']
        )

        model = get_model()
        metrics, cm, err = model.train_from_df(df, target_col)
        if err:
            raise HTTPException(status_code=400, detail=err)

        state = get_model_state()
        new_version = state['version'] + 1
        update_model_state(new_version, datetime.now(timezone.utc).isoformat(),
                           metrics, cm, model.class_names)

        clear_all()
        for _, row in df.iterrows():
            features = {f: (float(row[f]) if pd.api.types.is_numeric_dtype(df[f]) else str(row[f]))
                        for f in analysis['feature_names']}
            true_label = str(row[target_col])
            pred_class, probs, conf = model.predict(features)
            insert_record(features, true_label, pred_class, probs, conf)

        viz_list = generate_visualizations(df, target_col, analysis['feature_types'])
        viz_items = []
        for viz in viz_list:
            vid = save_visualization(upload_id, viz['viz_type'], viz['feature'],
                                    viz['title'], viz['image_base64'])
            viz_items.append(VisualizationItem(
                id=vid, viz_type=viz['viz_type'], feature_name=viz['feature'],
                title=viz['title'], image_base64=viz['image_base64']
            ))

        return CSVUploadResponse(
            upload_id=upload_id,
            filename=file.filename,
            n_rows=analysis['n_rows'],
            n_cols=analysis['n_cols'],
            feature_names=analysis['feature_names'],
            target_name=target_col,
            feature_types=analysis['feature_types'],
            class_distribution=analysis['class_distribution'],
            visualizations=viz_items,
            model_version=new_version,
            metrics=metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/csv-info")
def csv_info():
    meta = get_upload_meta()
    if not meta:
        return {"uploaded": False}
    return {"uploaded": True, **meta}


@app.get("/visualizations")
def list_visualizations(viz_type: str = None):
    vizs = get_visualizations()
    if viz_type:
        vizs = [v for v in vizs if v['viz_type'] == viz_type]
    return [VisualizationItem(**{k: v for k, v in viz.items() if k in VisualizationItem.__fields__}) for viz in vizs]


@app.delete("/csv")
def delete_csv():
    clear_uploads()
    clear_all()
    import model as m
    m._model_instance = None
    return {"status": "cleared"}
