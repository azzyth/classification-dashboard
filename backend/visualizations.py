"""Visualization generation for classification EDA."""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# Color palette for classes
CLASS_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316']


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def _setup_style():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['font.size'] = 10


def plot_segmented_bar(df: pd.DataFrame, feature: str, target: str) -> str:
    """Stacked bar showing class proportion per category."""
    _setup_style()
    fig, ax = plt.subplots()

    ct = pd.crosstab(df[feature], df[target], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=ax, color=CLASS_COLORS[:len(ct.columns)], edgecolor='white', linewidth=0.5)

    ax.set_title(f'Segmented Bar: {feature} by {target}', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Proportion')
    ax.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_density(df: pd.DataFrame, feature: str, target: str) -> str:
    """Overlapping KDE/histogram by class."""
    _setup_style()
    fig, ax = plt.subplots()

    classes = df[target].unique()
    for i, cls in enumerate(classes):
        subset = df[df[target] == cls][feature].dropna()
        if len(subset) > 1:
            subset.plot(kind='kde', ax=ax, label=str(cls), color=CLASS_COLORS[i % len(CLASS_COLORS)], linewidth=2)
        else:
            ax.hist(subset, bins=10, alpha=0.5, label=str(cls), color=CLASS_COLORS[i % len(CLASS_COLORS)])

    ax.set_title(f'Overlapping Density: {feature} by {target}', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend(title=target)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_point(df: pd.DataFrame, feature: str, target: str) -> str:
    """Mean +/- 95% CI per class (point plot)."""
    _setup_style()
    fig, ax = plt.subplots()

    classes = sorted(df[target].unique(), key=lambda x: str(x))
    means = []
    cis = []
    for cls in classes:
        vals = df[df[target] == cls][feature].dropna()
        m = vals.mean()
        se = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ci = 1.96 * se
        means.append(m)
        cis.append(ci)

    x_pos = range(len(classes))
    ax.errorbar(x_pos, means, yerr=cis, fmt='o-', color=CLASS_COLORS[0],
                capsize=5, capthick=2, markersize=8, linewidth=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_title(f'Point Plot: {feature} mean by {target}', fontsize=12, fontweight='bold')
    ax.set_xlabel(target)
    ax.set_ylabel(f'Mean {feature}')
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_crosstab_heatmap(df: pd.DataFrame, feature: str, target: str) -> str:
    """Heatmap of crosstab counts."""
    _setup_style()
    fig, ax = plt.subplots()

    ct = pd.crosstab(df[feature], df[target])
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5, cbar_kws={'label': 'Count'})

    ax.set_title(f'Crosstab Heatmap: {feature} x {target}', fontsize=12, fontweight='bold')
    ax.set_xlabel(target)
    ax.set_ylabel(feature)
    fig.tight_layout()
    return _fig_to_base64(fig)


def compute_woe(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """Compute WoE and IV for a feature."""
    # Bin numeric features
    if pd.api.types.is_numeric_dtype(df[feature]):
        try:
            bins = pd.qcut(df[feature], 5, duplicates='drop')
        except Exception:
            bins = pd.cut(df[feature], 5)
        bin_col = bins.astype(str)
    else:
        bin_col = df[feature].astype(str)

    d = pd.DataFrame({'bin': bin_col, 'y': df[target]})
    # For binary target: bad = positive class
    classes = sorted(df[target].unique())
    if len(classes) == 2:
        bad_class = classes[-1]  # assume last is positive
        d['is_bad'] = (d['y'] == bad_class).astype(int)
    else:
        # Multi-class: use first class as "bad" for WoE (simplified)
        d['is_bad'] = (d['y'] == classes[0]).astype(int)

    g = d.groupby('bin')['is_bad'].agg(['sum', 'count'])
    g['good'] = g['count'] - g['sum']
    g['dist_bad'] = g['sum'] / max(g['sum'].sum(), 1)
    g['dist_good'] = g['good'] / max(g['good'].sum(), 1)
    g['woe'] = np.log((g['dist_bad'] + 1e-6) / (g['dist_good'] + 1e-6))
    g['iv'] = (g['dist_bad'] - g['dist_good']) * g['woe']
    return g


def plot_woe(df: pd.DataFrame, feature: str, target: str) -> str:
    """WoE bar chart with IV annotation."""
    _setup_style()
    fig, ax = plt.subplots()

    woe_df = compute_woe(df, feature, target)
    iv_total = woe_df['iv'].sum()

    colors = ['#10b981' if w > 0 else '#ef4444' for w in woe_df['woe']]
    ax.bar(range(len(woe_df)), woe_df['woe'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(woe_df)))
    ax.set_xticklabels(woe_df.index, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'WoE: {feature} (IV={iv_total:.3f})', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{feature} (binned)')
    ax.set_ylabel('Weight of Evidence')
    fig.tight_layout()
    return _fig_to_base64(fig)


def detect_feature_type(series: pd.Series) -> str:
    """Auto-detect feature type: numeric, categorical, or binary."""
    n_unique = series.nunique(dropna=True)
    if n_unique <= 2:
        return 'binary'
    if pd.api.types.is_numeric_dtype(series):
        if n_unique > 10:
            return 'numeric'
        else:
            return 'categorical'
    return 'categorical'


def generate_visualizations(df: pd.DataFrame, target: str, feature_types: dict) -> list:
    """Generate all relevant visualizations for a dataset.

    Returns list of dicts: {viz_type, feature, title, image_base64}
    """
    results = []
    features = [c for c in df.columns if c != target]

    # Limit to 20 features for viz
    if len(features) > 20:
        # Sort by variance (numeric) or cardinality (categorical)
        variances = {}
        for f in features:
            if pd.api.types.is_numeric_dtype(df[f]):
                variances[f] = df[f].var()
            else:
                variances[f] = df[f].nunique()
        features = sorted(features, key=lambda f: variances.get(f, 0), reverse=True)[:20]

    for feat in features:
        ftype = feature_types.get(feat, 'numeric')

        if ftype == 'numeric':
            # Overlapping density + point plot
            try:
                img = plot_density(df, feat, target)
                results.append({'viz_type': 'density', 'feature': feat,
                                'title': f'Density: {feat}', 'image_base64': img})
            except Exception:
                pass
            try:
                img = plot_point(df, feat, target)
                results.append({'viz_type': 'point', 'feature': feat,
                                'title': f'Point: {feat}', 'image_base64': img})
            except Exception:
                pass
        else:
            # Segmented bar + crosstab heatmap
            try:
                img = plot_segmented_bar(df, feat, target)
                results.append({'viz_type': 'segmented_bar', 'feature': feat,
                                'title': f'Segmented Bar: {feat}', 'image_base64': img})
            except Exception:
                pass
            try:
                img = plot_crosstab_heatmap(df, feat, target)
                results.append({'viz_type': 'crosstab', 'feature': feat,
                                'title': f'Crosstab: {feat}', 'image_base64': img})
            except Exception:
                pass

        # WoE for all features
        try:
            img = plot_woe(df, feat, target)
            results.append({'viz_type': 'woe', 'feature': feat,
                            'title': f'WoE: {feat}', 'image_base64': img})
        except Exception:
            pass

    return results


def analyze_csv(df: pd.DataFrame) -> dict:
    """Analyze CSV: detect feature types, suggest target."""
    # Suggest target: last column if name contains target/label/class/y, else last column
    cols = list(df.columns)
    target_candidates = [c for c in cols if any(k in c.lower() for k in ['target', 'label', 'class', 'y', 'outcome', 'response'])]
    suggested_target = target_candidates[-1] if target_candidates else cols[-1]

    feature_types = {}
    for col in cols:
        if col == suggested_target:
            continue
        feature_types[col] = detect_feature_type(df[col])

    # Class distribution
    class_dist = df[suggested_target].value_counts().to_dict()

    return {
        'n_rows': len(df),
        'n_cols': len(cols),
        'feature_names': [c for c in cols if c != suggested_target],
        'target_name': suggested_target,
        'feature_types': feature_types,
        'class_distribution': {str(k): int(v) for k, v in class_dist.items()},
    }
