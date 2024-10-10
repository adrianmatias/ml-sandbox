from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.app.date_time import DateTime
from src.app.logger_custom import LoggerCustom
from src.app.schema_pydantic import Transaction, TransactionLabeled


class Schema:
    def __init__(self, df: DataFrame):
        self.y = "is_fraud"
        df = df.drop(columns=self.y)
        self.x_num = sorted(df.select_dtypes(include=[np.number]).columns)
        self.x_cat = sorted(df.drop(columns=self.x_num).columns)
        self.x = self.x_cat + self.x_num
        self.x_input_classifier: List[str] = []

        print(self.__dict__)


class ModelClassification:
    @staticmethod
    def predict_proba(X) -> float:
        return 0.9 if X["amount"] > 1000 else 0.1

    def get_feature_dict(self, transaction: Transaction) -> Dict:
        feature_dict = {
            "amount": transaction.amount,
            "transaction_type": transaction.transaction_type,
        }
        dt = DateTime().get_obj(dt_str=transaction.dtime)
        dt_dict = {
            "dt_year": dt.year,
            "dt_month": dt.month,
            "dt_day": dt.day,
            "dt_weekday": dt.weekday(),
            "dt_hour": dt.hour,
        }
        return {**feature_dict, **dt_dict, **self.parse_code(code=transaction.code)}

    def get_feature_and_label_dict(self, transaction: TransactionLabeled) -> Dict:
        return {
            **self.get_feature_dict(transaction=transaction),
            **{"is_fraud": transaction.is_fraud},
        }

    @staticmethod
    def parse_code(code: str):
        return dict(map(lambda x_i: (f"code_{x_i[0]}", x_i[1]), enumerate(code)))


class Evaluation(BaseModel):
    count: Optional[int]
    roc_auc: float
    log_loss: float
    brier_score_loss: float


class ModelCard(BaseModel):
    evaluation_dict: Dict[str, Evaluation]
    feat_importance: Dict[str, float]
    grid_search_cv_best_params: dict
    grid_search_cv_param_grid: dict


class TokenizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols: list, col_token: str):
        self.cat_cols = cat_cols
        self.col_token = col_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cat_cols)
        X[self.col_token] = X.apply(
            lambda row: " ".join([f"{col}|{val}" for col, val in row.items()]),
            axis=1,
        )
        return X[self.col_token].values


class LDATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.lda_model = LatentDirichletAllocation(n_components=self.n_components)
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.lda_model.fit(X)
        self.feature_names_ = [f"topic_{i}" for i in range(self.n_components)]
        return self

    def transform(self, X):
        lda_features = self.lda_model.transform(X)
        return lda_features


class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols
        self.model = IsolationForest(contamination="auto", random_state=42)

    def fit(self, X, y=None):
        if self.feature_cols is None:
            self.feature_cols = list(range(X.shape[1]))
        self.model.fit(X[:, self.feature_cols])
        return self

    def transform(self, X):
        anomaly_scores = self.model.decision_function(X[:, self.feature_cols]).reshape(
            -1, 1
        )
        return np.hstack([X, anomaly_scores])


class ModelClassificationCatBoost:
    def __init__(self):
        self.df: pd.DataFrame = None
        self.pipeline: Pipeline = None
        self.schema: Schema = None
        self.logger = LoggerCustom().logger

    def build_pipeline(self) -> Pipeline:
        col_token = "token_str"

        lda_pipeline = Pipeline(
            [
                (
                    "token_transformer",
                    TokenizerTransformer(
                        cat_cols=self.schema.x_cat, col_token=col_token
                    ),
                ),
                ("count_vectorizer", CountVectorizer(min_df=0.001)),
                ("lda", LDATransformer()),
            ]
        )

        feature_combiner = ColumnTransformer(
            [
                ("lda_features", lda_pipeline, self.schema.x_cat),
                ("num_features", "passthrough", self.schema.x_num),
            ]
        )

        combined_feature_pipeline = Pipeline(
            [
                ("combine_features", feature_combiner),
                ("isolation_forest", IsolationForestTransformer()),
            ]
        )
        pipeline = Pipeline(
            [
                ("combined_features", combined_feature_pipeline),
                (
                    "cat_boost_classifier",
                    CatBoostClassifier(
                        boosting_type="Plain",
                        loss_function="Logloss",
                        eval_metric="Logloss",
                        verbose=False,
                    ),
                ),
            ]
        )

        self.logger.info(f"Pipeline: {pipeline}")
        return pipeline

    def get_x_input_classifier(self):
        combined_features_pipeline = self.pipeline.named_steps["combined_features"]
        column_transformer = combined_features_pipeline.named_steps["combine_features"]

        x_input_classifier = []

        for name, transformer, columns in column_transformer.transformers:
            if name == "lda_features":
                lda_pipeline = transformer
                lda = lda_pipeline.named_steps["lda"]

                lda_feature_names = [f"topic_{i}" for i in range(lda.n_components)]
                x_input_classifier.extend(lda_feature_names)
            elif name == "num_features":
                x_input_classifier.extend(columns)

        x_input_classifier.append("anomaly_score")

        return x_input_classifier

    def _fit_cv(self, df: pd.DataFrame) -> GridSearchCV:
        self.schema = Schema(df=df)
        df = self.prepare_df(df=df)

        lda_n_components = "__".join(
            [
                "combined_features",
                "combine_features",
                "lda_features",
                "lda",
                "n_components",
            ]
        )
        grid_search_cv = GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid={
                lda_n_components: [
                    3,
                ],
                "cat_boost_classifier__iterations": [10],
                "cat_boost_classifier__depth": [3, 4],
            },
            scoring={
                "neg_log_loss": "neg_log_loss",
                "roc_auc": "roc_auc",
            },
            cv=2,
            refit="neg_log_loss",
            n_jobs=-1,
            verbose=1,
        )

        grid_search_cv.fit(X=df[self.schema.x], y=df[self.schema.y])

        self.pipeline = grid_search_cv.best_estimator_

        self.schema.x_input_classifier = self.get_x_input_classifier()

        return grid_search_cv

    def prepare_df(self, df: DataFrame) -> DataFrame:
        df[self.schema.x_cat] = df[self.schema.x_cat].astype(str).fillna("unknown")
        df[self.schema.x_num] = df[self.schema.x_num].fillna(-999999.9)
        return df

    def get_prob(self, df: DataFrame, pipeline: Pipeline) -> np.ndarray:
        df = self.prepare_df(df=df[self.schema.x])
        return pipeline.predict_proba(X=df)[:, 1]

    def get_evaluation(self, df: DataFrame) -> Evaluation:
        y_prob = self.get_prob(df=df, pipeline=self.pipeline)
        y_true = df[self.schema.y]

        evaluation = Evaluation(
            count=len(df),
            roc_auc=roc_auc_score(
                y_score=y_prob,
                y_true=y_true,
            ),
            log_loss=log_loss(
                y_pred=y_prob,
                y_true=y_true,
            ),
            brier_score_loss=brier_score_loss(
                y_prob=y_prob,
                y_true=y_true,
            ),
        )
        self.logger.info(f"evaluation: {evaluation}")
        return evaluation

    def get_evaluation_train_cv(self, grid_search_cv: GridSearchCV) -> Evaluation:
        index = grid_search_cv.best_index_
        result_dict = grid_search_cv.cv_results_

        evaluation = Evaluation(
            count=None,
            log_loss=-np.mean(result_dict["mean_test_neg_log_loss"][index]),
            roc_auc=np.mean(result_dict["mean_test_roc_auc"][index]),
            brier_score_loss=-1,
        )
        self.logger.info(f"evaluation: {evaluation}")
        return evaluation

    def get_feature_importance(self) -> Dict[str, float]:
        model_catboost = self.pipeline.steps[-1][-1]
        get_feature_importance = model_catboost.get_feature_importance()
        feature_importance_df = (
            DataFrame(
                {
                    "feature": self.schema.x_input_classifier,
                    "importance": get_feature_importance,
                }
            )
            .set_index("feature")
            .sort_values(by="importance", ascending=False)
        )
        feature_importance_df = feature_importance_df / feature_importance_df.sum()

        self.logger.info(f"feature_importance_df: {feature_importance_df.head(20)}")
        d = feature_importance_df.to_dict()["importance"]
        self.logger.info(f"d: {d}")
        return d

    def train_and_evaluate(self, df_train: DataFrame, df_test: DataFrame) -> ModelCard:
        self.schema = Schema(df=df_train)
        df_train = self.prepare_df(df=df_train)
        df_test = self.prepare_df(df=df_test)

        self.logger.info(f"Schema.x_cat: {self.schema.x_cat}")
        self.logger.info(f"Schema.x_num: {self.schema.x_num}")

        grid_search_cv = self._fit_cv(df=df_train)

        self.model = grid_search_cv.best_estimator_

        model_card = ModelCard(
            evaluation_dict={
                "train": self.get_evaluation(df=df_train),
                "test": self.get_evaluation(df=df_test),
                "train_cv": self.get_evaluation_train_cv(grid_search_cv=grid_search_cv),
            },
            feat_importance=self.get_feature_importance(),
            grid_search_cv_param_grid=grid_search_cv.param_grid,
            grid_search_cv_best_params=grid_search_cv.best_params_,
        )
        self.logger.info(f"model_card: {model_card}")
        return model_card
