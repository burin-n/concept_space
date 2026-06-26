import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from sklearn.metrics import log_loss


N_JOBS = int(os.environ.get("EVAL_N_JOBS", 8))
MAX_ITER = int(os.environ.get("EVAL_MAX_ITER", 200))


def train_classifier(args):
    if len(args) == 2:
        X, y = args
        model = "linear"
    elif len(args) == 3:
        X, y, model = args
        if model not in ["linear", "non-linear"]:
            raise ValueError(f"model should be 'linear' or 'non-linear', got {model}")
    else:
        raise ValueError("args should be a tuple of (X, y) or (X, y, model)")

    if model == "linear":
        clf_raw = LogisticRegression(n_jobs=8, max_iter=MAX_ITER, verbose=0, solver="lbfgs")
        clf_raw.fit(X, y)
    elif model == "non-linear":
        clf_raw = MLPClassifier(hidden_layer_sizes=(40,), random_state=1, solver='adam', activation='relu',
                                max_iter=MAX_ITER, verbose=True, early_stopping=True)
        clf_raw.fit(X, y)
    return clf_raw


def get_loss_val(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)
    test_loss = log_loss(y_test, y_pred_proba)
    return test_loss


def eval_linear_classifier_v3(proj, proj_name, proj_train_data, data_clean_set=None, data_other_set=None,
                               random_sample=True, rejection=True, predict_spk=False,
                               classifier_type="linear"):

    if data_clean_set is None:
        assert data_other_set is not None, "at least one data must not be None"
        num_data_subset_split = len(data_other_set)
    else:
        num_data_subset_split = len(data_clean_set)

    if num_data_subset_split == 6:
        if data_clean_set is not None:
            X_train_clean, y_phone_train_clean, y_spk_train_clean, X_test_clean, y_phone_test_clean, y_spk_test_clean = data_clean_set
        if data_other_set is not None:
            X_train_other, y_phone_train_other, y_spk_train_other, X_test_other, y_phone_test_other, y_spk_test_other = data_other_set
    elif num_data_subset_split == 4:
        if data_clean_set is not None:
            X_train_clean, y_train_clean, X_test_clean, y_test_clean = data_clean_set
        if data_other_set is not None:
            X_train_other, y_train_other, X_test_other, y_test_other = data_other_set

        if predict_spk:
            if data_clean_set is not None:
                y_spk_train_clean = y_train_clean
                y_spk_test_clean = y_test_clean
            if data_other_set is not None:
                y_spk_train_other = y_train_other
                y_spk_test_other = y_test_other
        else:
            if data_clean_set is not None:
                y_phone_train_clean = y_train_clean
                y_phone_test_clean = y_test_clean
            if data_other_set is not None:
                y_phone_train_other = y_train_other
                y_phone_test_other = y_test_other
    else:
        raise ValueError("data_clean_set and data_other_set should be a tuple of 6 or 4 elements")

    def gen_args_phone(proj, proj_name):
        if data_clean_set is None:
            testing_args = (
                (X_train_other, y_phone_train_other, proj, proj_name, "dev-other"),
                (X_test_other, y_phone_test_other, proj, proj_name, "test-other"),
            )
        elif data_other_set is None:
            testing_args = (
                (X_train_clean, y_phone_train_clean, proj, proj_name, "dev-clean"),
                (X_test_clean, y_phone_test_clean, proj, proj_name, "test-clean"),
            )
        else:
            testing_args = (
                (X_train_clean, y_phone_train_clean, proj, proj_name, "dev-clean"),
                (X_train_other, y_phone_train_other, proj, proj_name, "dev-other"),
                (np.concatenate([X_train_clean, X_train_other], axis=0),
                 np.concatenate([y_phone_train_clean, y_phone_train_other], axis=0),
                 proj, proj_name, "dev-all"),
                (X_test_clean, y_phone_test_clean, proj, proj_name, "test-clean"),
                (X_test_other, y_phone_test_other, proj, proj_name, "test-other"),
                (np.concatenate([X_test_clean, X_test_other], axis=0),
                 np.concatenate([y_phone_test_clean, y_phone_test_other], axis=0),
                 proj, proj_name, "test-all"),
            )
        return testing_args

    def gen_args_spk(proj, proj_name):
        if data_clean_set is None:
            testing_args = (
                (X_train_other, y_spk_train_other, proj, proj_name, "dev-other"),
                (X_test_other, y_spk_test_other, proj, proj_name, "test-other"),
            )
        elif data_other_set is None:
            testing_args = (
                (X_train_clean, y_spk_train_clean, proj, proj_name, "dev-clean"),
                (X_test_clean, y_spk_test_clean, proj, proj_name, "test-clean"),
            )
        else:
            testing_args = (
                (X_train_clean, y_spk_train_clean, proj, proj_name, "dev-clean"),
                (X_train_other, y_spk_train_other, proj, proj_name, "dev-other"),
                (np.concatenate([X_train_clean, X_train_other], axis=0),
                 np.concatenate([y_spk_train_clean, y_spk_train_other], axis=0),
                 proj, proj_name, "dev-all"),
                (X_test_clean, y_spk_test_clean, proj, proj_name, "test-clean"),
                (X_test_other, y_spk_test_other, proj, proj_name, "test-other"),
                (np.concatenate([X_test_clean, X_test_other], axis=0),
                 np.concatenate([y_spk_test_clean, y_spk_test_other], axis=0),
                 proj, proj_name, "test-all"),
            )
        return testing_args

    def eval_v3_(train_data_X, train_data_Y, proj, proj_name, train_data_name, random_sample=True, rejection=False, predict_spk=False):
        score = dict()
        portions = [1]
        score["proj name"] = [proj_name] * len(portions)
        score["clf_train_data"] = [train_data_name] * len(portions)
        score["clf_train_portion"] = portions
        if rejection:
            fn = proj.rejection
        else:
            fn = proj.projection

        if random_sample:
            sample_random_total = np.random.choice(len(train_data_X), len(train_data_X), replace=False)
        else:
            sample_random_total = np.arange(len(train_data_X))

        for portion in portions:
            sample_n = int(len(train_data_X) * portion)
            sample_choice = sample_random_total[:sample_n]
            clf_rect = train_classifier((fn(np.concatenate([train_data_X[sample_choice]], axis=0)),
                                         np.concatenate([train_data_Y[sample_choice]], axis=0), classifier_type))
            if predict_spk:
                if data_clean_set is not None:
                    score.setdefault("dev-clean", []).append(clf_rect.score(fn(X_train_clean), y_spk_train_clean))
                    score.setdefault("test-clean", []).append(clf_rect.score(fn(X_test_clean), y_spk_test_clean))
                    score.setdefault("dev-clean-loss", []).append(get_loss_val(clf_rect, fn(X_train_clean), y_spk_train_clean))
                    score.setdefault("test-clean-loss", []).append(get_loss_val(clf_rect, fn(X_test_clean), y_spk_test_clean))
                if data_other_set is not None:
                    score.setdefault("dev-other", []).append(clf_rect.score(fn(X_train_other), y_spk_train_other))
                    score.setdefault("test-other", []).append(clf_rect.score(fn(X_test_other), y_spk_test_other))
                    score.setdefault("dev-clean-loss", []).append(get_loss_val(clf_rect, fn(X_train_other), y_spk_train_other))
                    score.setdefault("test-clean-loss", []).append(get_loss_val(clf_rect, fn(X_test_other), y_spk_test_other))
            else:
                if data_clean_set is not None:
                    score.setdefault("dev-clean", []).append(clf_rect.score(fn(X_train_clean), y_phone_train_clean))
                    score.setdefault("test-clean", []).append(clf_rect.score(fn(X_test_clean), y_phone_test_clean))
                    score.setdefault("dev-clean-loss", []).append(get_loss_val(clf_rect, fn(X_train_clean), y_phone_train_clean))
                    score.setdefault("test-clean-loss", []).append(get_loss_val(clf_rect, fn(X_test_clean), y_phone_test_clean))
                if data_other_set is not None:
                    score.setdefault("dev-other", []).append(clf_rect.score(fn(X_train_other), y_phone_train_other))
                    score.setdefault("test-other", []).append(clf_rect.score(fn(X_test_other), y_phone_test_other))
                    score.setdefault("dev-clean-loss", []).append(get_loss_val(clf_rect, fn(X_train_other), y_phone_train_other))
                    score.setdefault("test-clean-loss", []).append(get_loss_val(clf_rect, fn(X_test_other), y_phone_test_other))

        return pd.DataFrame(score)

    if predict_spk:
        testing_args = gen_args_spk(proj, proj_name)
    else:
        testing_args = gen_args_phone(proj, proj_name)

    results = Parallel(n_jobs=len(testing_args))(
        delayed(eval_v3_)(*args, random_sample=random_sample, rejection=rejection, predict_spk=predict_spk)
        for args in testing_args
    )
    score_v3 = pd.concat(results)
    col_names = [n_ for n_ in score_v3.columns if n_ != "proj name"]
    score_v3["proj_train_data"] = [proj_train_data] * len(score_v3)
    col_names = ["proj name", "proj_train_data"] + col_names
    score_v3 = score_v3[col_names]
    return score_v3


def eval_v3_wrapper(projection_to_use, proj_name, proj_train_data, data_clean_set, data_other_set,
                    save_folder="scores", suffix_y1="", suffix_y2="-spk",
                    classifier_type="linear", overwrite=False):
    assert classifier_type in ["linear", "non-linear"], f"classifier_type should be 'linear' or 'non-linear', got {classifier_type}"
    for predict_y2 in [False, True]:
        for mode in ["rejection", "projection"]:
            suffix = suffix_y2 if predict_y2 else suffix_y1
            save_path = os.path.join(save_folder, proj_name, mode, f"{proj_train_data}{suffix}.csv")
            if not overwrite and os.path.exists(save_path):
                print(f"Skip {save_path} because it already exists")
                continue

            score = eval_linear_classifier_v3(projection_to_use, proj_name, proj_train_data, data_clean_set, data_other_set,
                                              random_sample=True,
                                              rejection=True if mode == "rejection" else False,
                                              predict_spk=predict_y2,
                                              classifier_type=classifier_type)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            score.to_csv(save_path, index=False)
            print(save_path)
