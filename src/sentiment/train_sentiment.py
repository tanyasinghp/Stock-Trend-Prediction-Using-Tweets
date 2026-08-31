"""Train and select a finance-tweet sentiment classifier.

Trains on the 5,791 hand-labeled tweets carried over from the original
project (data/raw/labeled_tweets.csv). VADER is retained as an unsupervised
baseline. The winning model is selected on cross-validated ROC-AUC computed
on the TRAIN split only; the test split is touched exactly once, at the end.

Outputs:
    models/sentiment_model.joblib
    results/sentiment/model_comparison.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sentiment.finance_text import Preprocess_Tweets  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEED = 42

# Lower rank == simpler model. Used only as a tie-break, fixed in advance.
COMPLEXITY = {
    "TF-IDF + MultinomialNB": 0,
    "TF-IDF + LogisticRegression": 1,
    "TF-IDF + LinearSVC (calibrated)": 2,
}


def load_labeled() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/raw/labeled_tweets.csv")
    df = Preprocess_Tweets(df)
    df = df.dropna(subset=["Text_Cleaned"])
    df = df[df["Text_Cleaned"].str.strip().str.len() > 0].copy()
    # original labels are {-1, 1}; map to {0, 1}
    df["y"] = (df["Sentiment"] > 0).astype(int)
    return df.reset_index(drop=True)


def vader_baseline(texts: pd.Series, y: np.ndarray) -> dict:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sid = SentimentIntensityAnalyzer()
    scores = texts.apply(lambda t: sid.polarity_scores(t)["compound"]).to_numpy()
    pred = (scores >= 0).astype(int)
    return {
        "model": "VADER (unsupervised baseline)",
        "cv_roc_auc": np.nan,
        "test_accuracy": accuracy_score(y, pred),
        "test_f1": f1_score(y, pred),
        "test_roc_auc": roc_auc_score(y, scores),
    }


def candidates() -> dict[str, Pipeline]:
    tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "TF-IDF + MultinomialNB": Pipeline(
            [("tfidf", tfidf()), ("clf", MultinomialNB(alpha=1.0))]
        ),
        "TF-IDF + LogisticRegression": Pipeline(
            [("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))]
        ),
        "TF-IDF + LinearSVC (calibrated)": Pipeline(
            [
                ("tfidf", tfidf()),
                ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, random_state=SEED), cv=3)),
            ]
        ),
    }


def main() -> None:
    df = load_labeled()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["Text_Cleaned"], df["y"], test_size=0.2, stratify=df["y"], random_state=SEED
    )
    print(f"labeled tweets: {len(df)}  train={len(X_tr)}  test={len(X_te)}  pos_rate={df.y.mean():.3f}")

    rows = [vader_baseline(X_te, y_te.to_numpy())]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted = {}
    for name, pipe in candidates().items():
        folds = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        auc_cv, auc_se = folds.mean(), folds.std(ddof=1) / np.sqrt(len(folds))
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "cv_roc_auc": auc_cv,
                "cv_roc_auc_se": auc_se,
                "complexity_rank": COMPLEXITY[name],
                "test_accuracy": accuracy_score(y_te, pred),
                "test_f1": f1_score(y_te, pred),
                "test_roc_auc": roc_auc_score(y_te, proba),
            }
        )
        fitted[name] = pipe
        print(f"  {name:34s} cv_auc={auc_cv:.4f} test_auc={rows[-1]['test_roc_auc']:.4f}")

    res = pd.DataFrame(rows)
    (ROOT / "results/sentiment").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "results/sentiment/model_comparison.csv", index=False)

    # Pre-registered selection rule (one-standard-error rule):
    #   1. find the highest CV ROC-AUC among supervised candidates
    #   2. keep every model within 1 SE of it (statistically indistinguishable)
    #   3. among those, choose the LEAST complex
    # Decided before looking at any test score. Test split is used for
    # reporting only, never for selection.
    sup = res.dropna(subset=["cv_roc_auc"]).copy()
    top = sup.loc[sup["cv_roc_auc"].idxmax()]
    threshold = top["cv_roc_auc"] - top["cv_roc_auc_se"]
    within = sup[sup["cv_roc_auc"] >= threshold]
    best_name = within.loc[within["complexity_rank"].idxmin(), "model"]
    print(f"\ntop CV AUC          : {top['model']} ({top['cv_roc_auc']:.4f} +/- {top['cv_roc_auc_se']:.4f})")
    print(f"within 1 SE         : {list(within['model'])}")
    print(f"selected (simplest) : {best_name}")

    # Refit winner on ALL labeled data before scoring StockNet.
    final = candidates()[best_name]
    final.fit(df["Text_Cleaned"], df["y"])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models/sentiment_model.joblib")
    (ROOT / "results/sentiment/selected_model.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "selection_criterion": (
                    "one-standard-error rule on 5-fold CV ROC-AUC (train split only); "
                    "among models within 1 SE of the best, the least complex is chosen"
                ),
                "top_cv_model": top["model"],
                "top_cv_roc_auc": round(float(top["cv_roc_auc"]), 4),
                "one_se_threshold": round(float(threshold), 4),
                "models_within_1se": list(within["model"]),
                "n_labeled": int(len(df)),
                "refit_on": "all labeled data",
            },
            indent=2,
        )
    )
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
