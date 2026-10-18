"""Classification model wrapper: train, predict, metrics, confusion matrix."""
import json
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "encoder.joblib")
ORDINAL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ordinal_encoder.joblib")


class ClassificationModel:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.ordinal_encoder = None
        self.feature_names = []
        self.class_names = []
        self.target_name = None
        self.cat_cols = []
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
            self.class_names = list(self.label_encoder.classes_)
            fn_path = os.path.join(os.path.dirname(MODEL_PATH), "features.json")
            if os.path.exists(fn_path):
                with open(fn_path) as f:
                    self.feature_names = json.load(f)
            tn_path = os.path.join(os.path.dirname(MODEL_PATH), "target.json")
            if os.path.exists(tn_path):
                with open(tn_path) as f:
                    self.target_name = json.load(f)
            if os.path.exists(ORDINAL_ENCODER_PATH):
                self.ordinal_encoder = joblib.load(ORDINAL_ENCODER_PATH)
                cc_path = os.path.join(os.path.dirname(MODEL_PATH), "cat_cols.json")
                if os.path.exists(cc_path):
                    with open(cc_path) as f:
                        self.cat_cols = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.label_encoder, ENCODER_PATH)
        if self.ordinal_encoder is not None:
            joblib.dump(self.ordinal_encoder, ORDINAL_ENCODER_PATH)
        with open(os.path.join(os.path.dirname(MODEL_PATH), "features.json"), "w") as f:
            json.dump(self.feature_names, f)
        with open(os.path.join(os.path.dirname(MODEL_PATH), "target.json"), "w") as f:
            json.dump(self.target_name, f)
        with open(os.path.join(os.path.dirname(MODEL_PATH), "cat_cols.json"), "w") as f:
            json.dump(self.cat_cols, f)

    def set_schema(self, feature_names, class_names=None, target_name=None):
        self.feature_names = list(feature_names)
        if class_names:
            self.class_names = list(class_names)
        if target_name:
            self.target_name = target_name

    def _to_matrix(self, df, feature_names=None):
        fn = feature_names or self.feature_names
        X = df[fn].values if isinstance(df, pd.DataFrame) else np.array([[r.get(f, 0.0) for f in fn] for r in df])
        return X

    def train_from_df(self, df: pd.DataFrame, target_col: str):
        """Train on a DataFrame. Returns (metrics, confusion_matrix, error)."""
        if target_col not in df.columns:
            return None, None, f"Target column '{target_col}' not found"

        self.feature_names = [c for c in df.columns if c != target_col]
        self.target_name = target_col

        X = df[self.feature_names].values
        y_raw = df[target_col].values

        # Handle missing values
        for i in range(X.shape[1]):
            col = X[:, i]
            if pd.isna(col).any():
                if np.issubdtype(col.dtype, np.number):
                    X[:, i] = np.where(pd.isna(col), np.nanmedian(col), col)
                else:
                    # Categorical: fill with mode
                    from collections import Counter
                    mode_val = Counter(col[~pd.isna(col)]).most_common(1)[0][0] if len(col[~pd.isna(col)]) > 0 else 0
                    X[:, i] = np.where(pd.isna(col), mode_val, col)

        # Encode categorical features as numeric
        from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
        cat_cols = []
        for i, f in enumerate(self.feature_names):
            if not np.issubdtype(X[:, i].dtype, np.number):
                cat_cols.append(i)
        self.cat_cols = cat_cols
        if cat_cols:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[:, cat_cols] = oe.fit_transform(X[:, cat_cols])
            self.ordinal_encoder = oe
        else:
            self.ordinal_encoder = None

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw.astype(str))
        self.class_names = list(self.label_encoder.classes_)
        n_classes = len(self.class_names)

        if n_classes < 2:
            return None, None, "Need at least 2 distinct classes"

        classes, counts = np.unique(y, return_counts=True)
        n = len(y)
        c = len(classes)
        weights = n / (c * counts)
        weights = np.minimum(weights, 3.0)
        cw = {int(cls): float(w) for cls, w in zip(classes, weights)}

        min_count = int(counts.min())
        n_splits = min(5, min_count)
        if n_splits < 2:
            n_splits = 2
        n_splits = min(n_splits, len(y))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y), dtype=int)
        oof_probs = np.zeros((len(y), n_classes))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            tr_classes, tr_counts = np.unique(y_tr, return_counts=True)
            tr_n = len(y_tr)
            tr_c = len(tr_classes)
            tr_weights = tr_n / (tr_c * tr_counts)
            tr_weights = np.minimum(tr_weights, 3.0)
            tr_cw = {int(cls): float(w) for cls, w in zip(tr_classes, tr_weights)}

            clf = RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight=tr_cw,
                random_state=42, n_jobs=-1
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)
            full_probs = np.zeros((len(val_idx), n_classes))
            for j, cls in enumerate(clf.classes_):
                full_probs[:, int(cls)] = probs[:, j]
            preds = full_probs.argmax(axis=1)
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = full_probs

        final_clf = RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight=cw,
            random_state=42, n_jobs=-1
        )
        final_clf.fit(X, y)
        self.model = final_clf
        self._save()

        metrics = self._compute_metrics_from_arrays(y, oof_preds, oof_probs, n_classes)
        cm = confusion_matrix(y, oof_preds, labels=list(range(n_classes)))

        return metrics, cm.tolist(), None

    def predict(self, features: dict):
        if self.model is None or not self.feature_names:
            n = len(self.class_names) if self.class_names else 3
            probs = [1.0 / n] * n
            return (self.class_names[0] if self.class_names else 'A'), probs, 1.0 / n

        # Build feature vector, encoding categoricals
        X = np.zeros((1, len(self.feature_names)))
        for i, f in enumerate(self.feature_names):
            val = features.get(f, 0.0)
            X[0, i] = val if not isinstance(val, str) else 0.0

        # Apply ordinal encoding for categorical columns
        if self.ordinal_encoder is not None and self.cat_cols:
            cat_vals = np.array([[features.get(self.feature_names[i], '') for i in self.cat_cols]])
            try:
                encoded = self.ordinal_encoder.transform(cat_vals)
                for idx, col_i in enumerate(self.cat_cols):
                    X[0, col_i] = encoded[0, idx]
            except Exception:
                pass

        X = X.astype(float)

        try:
            probs = self.model.predict_proba(X)[0]
        except Exception:
            n = len(self.class_names) if self.class_names else 3
            probs = [1.0 / n] * n
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
        confidence = float(probs[pred_idx])
        return pred_class, probs.tolist(), confidence

    def _compute_metrics_from_arrays(self, y_true, y_pred, y_prob, n_classes):
        acc = float(accuracy_score(y_true, y_pred))
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

        per_class = {}
        for i, cls in enumerate(self.class_names):
            tp = int(((y_pred == i) & (y_true == i)).sum())
            fp = int(((y_pred == i) & (y_true != i)).sum())
            fn = int(((y_pred != i) & (y_true == i)).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            per_class[cls] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": int((y_true == i).sum()),
            }

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro, 4),
            "micro_f1": round(micro, 4),
            "per_class": per_class,
        }

    def compute_metrics(self, records):
        labeled = [r for r in records if r["true_label"] is not None and r["predicted_label"] is not None]
        if not labeled:
            return None
        # Use stored predictions
        y_true_str = np.array([r["true_label"] for r in labeled])
        y_pred_str = np.array([r["predicted_label"] for r in labeled])

        try:
            y_true = self.label_encoder.transform(y_true_str) if self.label_encoder else y_true_str
            y_pred = self.label_encoder.transform(y_pred_str) if self.label_encoder else y_pred_str
        except Exception:
            name_to_idx = {name: i for i, name in enumerate(self.class_names)}
            y_true = np.array([name_to_idx.get(str(l), 0) for l in y_true_str])
            y_pred = np.array([name_to_idx.get(str(l), 0) for l in y_pred_str])

        n_classes = len(self.class_names)
        oof_preds = y_pred
        oof_probs = np.array([r["probabilities"] for r in labeled])
        if oof_probs.shape[1] != n_classes:
            oof_probs = np.zeros((len(labeled), n_classes))

        return self._compute_metrics_from_arrays(y_true, oof_preds, oof_probs, n_classes)

    def compute_confusion_matrix(self, records):
        labeled = [r for r in records if r["true_label"] is not None and r["predicted_label"] is not None]
        if not labeled:
            return None, None
        y_true_str = np.array([r["true_label"] for r in labeled])
        y_pred_str = np.array([r["predicted_label"] for r in labeled])

        try:
            y_true = self.label_encoder.transform(y_true_str) if self.label_encoder else y_true_str
            y_pred = self.label_encoder.transform(y_pred_str) if self.label_encoder else y_pred_str
        except Exception:
            name_to_idx = {name: i for i, name in enumerate(self.class_names)}
            y_true = np.array([name_to_idx.get(str(l), 0) for l in y_true_str])
            y_pred = np.array([name_to_idx.get(str(l), 0) for l in y_pred_str])

        n_classes = len(self.class_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        per_class_f1 = {}
        for i, cls in enumerate(self.class_names):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            per_class_f1[cls] = round(f1, 4)

        return cm.tolist(), per_class_f1

    def get_distribution(self, records):
        labeled = [r for r in records if r["true_label"] is not None]
        if not labeled or not self.class_names:
            n = len(self.class_names) if self.class_names else 3
            return [1.0 / n] * n, [0.0] * n

        y_true_str = np.array([r["true_label"] for r in labeled])
        y_pred_str = np.array([r["predicted_label"] for r in labeled])

        try:
            y_true = self.label_encoder.transform(y_true_str) if self.label_encoder else y_true_str
            y_pred = self.label_encoder.transform(y_pred_str) if self.label_encoder else y_pred_str
        except Exception:
            name_to_idx = {name: i for i, name in enumerate(self.class_names)}
            y_true = np.array([name_to_idx.get(str(l), 0) for l in y_true_str])
            y_pred = np.array([name_to_idx.get(str(l), 0) for l in y_pred_str])

        n = len(y_true)
        n_classes = len(self.class_names)
        prior = [float((y_true == i).sum() / n) for i in range(n_classes)]
        predicted = [float((y_pred == i).sum() / n) for i in range(n_classes)]
        return prior, predicted


_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = ClassificationModel()
    return _model_instance
