"""SQLite accumulating data store for classification records."""
import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            features TEXT NOT NULL,
            true_label TEXT,
            predicted_label TEXT,
            probabilities TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL DEFAULT 0,
            trained_at TEXT,
            metrics TEXT,
            confusion_matrix TEXT,
            class_names TEXT
        );
        INSERT OR IGNORE INTO model_state (id, version) VALUES (1, 0);

        CREATE TABLE IF NOT EXISTS csv_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            n_rows INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            feature_names TEXT NOT NULL,
            target_name TEXT NOT NULL,
            feature_types TEXT NOT NULL,
            class_distribution TEXT,
            uploaded_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS visualizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            viz_type TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            title TEXT NOT NULL,
            image_base64 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (upload_id) REFERENCES csv_uploads(id)
        );
    """)
    conn.commit()
    conn.close()


def save_upload_meta(filename, n_rows, n_cols, feature_names, target_name, feature_types, class_distribution):
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO csv_uploads (filename, n_rows, n_cols, feature_names, target_name, feature_types, class_distribution, uploaded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (filename, n_rows, n_cols, json.dumps(feature_names), target_name,
         json.dumps(feature_types), json.dumps(class_distribution),
         datetime.now(timezone.utc).isoformat()))
    uid = cur.lastrowid
    conn.commit()
    conn.close()
    return uid


def get_upload_meta():
    conn = _conn()
    row = conn.execute("SELECT * FROM csv_uploads ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row['id'],
        'filename': row['filename'],
        'n_rows': row['n_rows'],
        'n_cols': row['n_cols'],
        'feature_names': json.loads(row['feature_names']),
        'target_name': row['target_name'],
        'feature_types': json.loads(row['feature_types']),
        'class_distribution': json.loads(row['class_distribution']) if row['class_distribution'] else {},
        'uploaded_at': row['uploaded_at'],
    }


def save_visualization(upload_id, viz_type, feature_name, title, image_base64):
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO visualizations (upload_id, viz_type, feature_name, title, image_base64, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (upload_id, viz_type, feature_name, title, image_base64, datetime.now(timezone.utc).isoformat()))
    vid = cur.lastrowid
    conn.commit()
    conn.close()
    return vid


def get_visualizations(upload_id=None):
    conn = _conn()
    if upload_id:
        rows = conn.execute(
            "SELECT * FROM visualizations WHERE upload_id=? ORDER BY id", (upload_id,)).fetchall()
    else:
        # Get latest upload's viz
        latest = conn.execute("SELECT id FROM csv_uploads ORDER BY id DESC LIMIT 1").fetchone()
        if latest:
            rows = conn.execute(
                "SELECT * FROM visualizations WHERE upload_id=? ORDER BY id", (latest['id'],)).fetchall()
        else:
            rows = []
    conn.close()
    return [{
        'id': r['id'],
        'upload_id': r['upload_id'],
        'viz_type': r['viz_type'],
        'feature_name': r['feature_name'],
        'title': r['title'],
        'image_base64': r['image_base64'],
        'created_at': r['created_at'],
    } for r in rows]


def clear_uploads():
    conn = _conn()
    conn.execute("DELETE FROM visualizations")
    conn.execute("DELETE FROM csv_uploads")
    conn.commit()
    conn.close()


def insert_record(features: dict, true_label: Optional[str], predicted_label: str,
                  probabilities: list, confidence: float) -> int:
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO records (features, true_label, predicted_label, probabilities, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (json.dumps(features), true_label, predicted_label, json.dumps(probabilities),
         confidence, datetime.now(timezone.utc).isoformat()))
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid


def get_all_records(limit: int = 500, offset: int = 0) -> list:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM records ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "features": json.loads(r["features"]),
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "probabilities": json.loads(r["probabilities"]) if r["probabilities"] else [],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        })
    return result


def get_record_count() -> int:
    conn = _conn()
    n = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    conn.close()
    return n


def get_labeled_records() -> list:
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM records WHERE true_label IS NOT NULL").fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "features": json.loads(r["features"]),
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "probabilities": json.loads(r["probabilities"]) if r["probabilities"] else [],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        })
    return result


def clear_all():
    conn = _conn()
    conn.execute("DELETE FROM records")
    conn.execute("UPDATE model_state SET version=0, trained_at=NULL, metrics=NULL, confusion_matrix=NULL, class_names=NULL")
    conn.commit()
    conn.close()


def get_model_state() -> dict:
    conn = _conn()
    row = conn.execute("SELECT * FROM model_state WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {"version": 0, "trained_at": None, "metrics": None, "confusion_matrix": None, "class_names": None}
    return {
        "version": row["version"],
        "trained_at": row["trained_at"],
        "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
        "confusion_matrix": json.loads(row["confusion_matrix"]) if row["confusion_matrix"] else None,
        "class_names": json.loads(row["class_names"]) if row["class_names"] else None,
    }


def update_model_state(version: int, trained_at: str, metrics: dict,
                       confusion_matrix: list, class_names: list):
    conn = _conn()
    conn.execute(
        "UPDATE model_state SET version=?, trained_at=?, metrics=?, confusion_matrix=?, class_names=? WHERE id=1",
        (version, trained_at, json.dumps(metrics), json.dumps(confusion_matrix), json.dumps(class_names)))
    conn.commit()
    conn.close()
