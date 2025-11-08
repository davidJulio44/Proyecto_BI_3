from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def _prepare_X(df: pd.DataFrame, features: list[str]):
    X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    return X.values, X.index

def kmeans_cluster(df: pd.DataFrame, features: list[str], n_clusters: int = 3, random_state: int = 42):
    X, index = _prepare_X(df, features)
    pipe = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state))])
    labels = pipe.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels))>1 else np.nan
    return pd.DataFrame({'label': labels}, index=index), pipe, sil

def dbscan_cluster(df: pd.DataFrame, features: list[str], eps: float = 0.5, min_samples: int = 5):
    X, index = _prepare_X(df, features)
    pipe = Pipeline([('scaler', StandardScaler()), ('dbscan', DBSCAN(eps=eps, min_samples=min_samples))])
    labels = pipe.fit_predict(X)
    valid = labels != -1
    sil = silhouette_score(X[valid], labels[valid]) if valid.sum()>1 and len(set(labels[valid]))>1 else np.nan
    return pd.DataFrame({'label': labels}, index=index), pipe, sil

def agglomerative_cluster(df: pd.DataFrame, features: list[str], n_clusters: int = 3, linkage: str = 'ward'):
    X, index = _prepare_X(df, features)
    pipe = Pipeline([('scaler', StandardScaler()), ('agg', AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage))])
    labels = pipe.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels))>1 else np.nan
    return pd.DataFrame({'label': labels}, index=index), pipe, sil
