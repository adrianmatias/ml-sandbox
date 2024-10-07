from typing import Dict, Optional

import numpy as np
from catboost import CatBoostClassifier
from pandas import DataFrame
from pydantic import BaseModel
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

from fraud.app.date_time import DateTime
from fraud.app.logger_custom import LoggerCustom
from fraud.app.schema_pydantic import Transaction, TransactionLabeled


class Schema:
    def __init__(self, df: DataFrame):
        self.y = "is_fraud"
        df = df.drop(columns=self.y)
        self.x_num = sorted(df.select_dtypes(include=[np.number]).columns)
        self.x_cat = sorted(df.drop(columns=self.x_num).columns)
        self.x = self.x_cat + self.x_num
        self.x_cat_index_list = list(range(len(self.x_cat)))

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


class ModelClassificationCatBoost(ModelClassification):
    def __init__(self):
        self.df: DataFrame = None
        self.model: CatBoostClassifier = None
        self.schema: Schema = None
        self.logger = LoggerCustom().logger

    def _fit_cv(self, df: DataFrame) -> GridSearchCV:
        self.schema = Schema(df=df)
        df = self.prepare_df(df=df)

        model = CatBoostClassifier(
            boosting_type="Plain",
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
        )
        grid_search_cv = GridSearchCV(
            estimator=model,
            param_grid={
                "iterations": [5],
                "depth": [3, 4],
                "learning_rate": [0.3],
                "l2_leaf_reg": [6],
                "auto_class_weights": ["SqrtBalanced"],
            },
            scoring={
                "neg_log_loss": "neg_log_loss",
                "roc_auc": "roc_auc",
            },
            cv=4,
            refit="neg_log_loss",
            n_jobs=-1,
            verbose=1,
        )
        grid_search_cv.fit(
            X=df[self.schema.x],
            y=df[self.schema.y],
            cat_features=self.schema.x_cat_index_list,
        )

        self.cat_model = grid_search_cv.best_estimator_

        return grid_search_cv

    def prepare_df(self, df: DataFrame) -> DataFrame:
        df[self.schema.x_cat] = df[self.schema.x_cat].astype(str).fillna("unknown")
        df[self.schema.x_num] = df[self.schema.x_num].fillna(-999999.9)
        return df

    def get_prob(self, df: DataFrame, cat_model: CatBoostClassifier) -> np.ndarray:
        df = self.prepare_df(df=df[self.schema.x])
        return cat_model.predict_proba(X=df, thread_count=1)[:, 1]

    def get_evaluation(self, df: DataFrame) -> Evaluation:
        y_prob = self.get_prob(df=df, cat_model=self.cat_model)
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
        feature_importance_df = (
            DataFrame(
                {
                    "feature": self.schema.x,
                    "importance": self.cat_model.get_feature_importance(),
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
