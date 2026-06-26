from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

from concept_space.LEACE import LeaceEraser_v2, LeaceFitter_v2
from concept_space.projector import EYE_proj, LinearCLF_proj, PCA_proj, LDA_proj, LEACE_proj, _base_projection
from concept_space.utils import get_mean_agg

supported_estimators_ = [
    "LEACE",
    "COV",
    "Linear",
    "MLR", # redirect to Linear
    "PCA",
    "CPCA", # redirect to PCA
    "LDA",
    "RANDOM",
    "EYE",
]


def get_estimator_by_name(name):
    if name not in supported_estimators_:
        raise ValueError(f"Estimator {name} is not supported. Supported: {supported_estimators_}")
    return globals()[f"get_{name}_model"]


def serialise_label(y):
    if type(y) == list:
        y = np.asarray(y)
    onehot_encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    return onehot_encoder, onehot_encoder.transform(y.reshape(-1, 1))


def get_COV_model(*args, **kwargs):
    if "model_y1_suffix" not in kwargs:
        kwargs["model_y1_suffix"] = "COV_phone"
    if "model_y2_suffix" not in kwargs:
        kwargs["model_y2_suffix"] = "COV_spk"
    return get_LEACE_model(*args, **kwargs, method="orth")


def get_LEACE_model(data_train_=None, y1_train_=None, y2_train_=None, dim=None, leace_use_whitening=True,
                    add_noise=False, model_suffix="", model_y1_suffix=None, model_y2_suffix=None, method=None,
                    load_from=None, X=None, y1=None, y2=None):

    if data_train_ is None and X is not None:
        data_train_ = X
    if y1_train_ is None and y1 is not None:
        y1_train_ = y1
    if y2_train_ is None and y2 is not None:
        y2_train_ = y2

    if load_from is not None:
        print(f"Loading LEACE model from {load_from}")
        with open(f"{load_from}{model_y1_suffix}.pk", "rb") as in_f:
            model_y1 = pickle.load(in_f)
        with open(f"{load_from}{model_y2_suffix}.pk", "rb") as in_f:
            model_y2 = pickle.load(in_f)
        if not isinstance(model_y1, LEACE_proj):
            model_y1 = LEACE_proj(model_y1, model_y1_suffix, n_components=dim)
        if not isinstance(model_y2, LEACE_proj):
            model_y2 = LEACE_proj(model_y2, model_y2_suffix, n_components=dim)
        return model_y1, model_y2

    if method is None:
        method = "orth" if leace_use_whitening == False else "leace"
        print(f"method is not provided. Using method {method} based on leace_use_whitening={leace_use_whitening}")
    else:
        print(f"Using method {method} for LEACE model fitting")

    if type(data_train_) is not torch.Tensor:
        data_train_ = torch.from_numpy(data_train_)
    y1_onehot_ = torch.from_numpy(OneHotEncoder(sparse_output=False).fit_transform(y1_train_.reshape(-1, 1)))

    model_y1 = LeaceEraser_v2.fit(data_train_, y1_onehot_, method=method, svd_tol=1e-7)
    if model_y1_suffix is None:
        model_y1_suffix = f"LEACE_phone{model_suffix}"
    if model_y2_suffix is None:
        model_y2_suffix = f"LEACE_spk{model_suffix}"

    if y2_train_ is not None:
        y2_onehot_ = torch.from_numpy(OneHotEncoder(sparse_output=False).fit_transform(y2_train_.reshape(-1, 1)))
        model_y2 = LeaceEraser_v2.fit(data_train_, y2_onehot_, method=method, svd_tol=1e-7)
        return LEACE_proj(model_y1, model_y1_suffix, n_components=dim), LEACE_proj(model_y2, model_y2_suffix, n_components=dim)
    else:
        return LEACE_proj(model_y1, model_y1_suffix, n_components=dim)


def get_random_model(in_dim=768, out_dim=39, name="RANDOM"):
    R = np.random.normal(0, 1 / np.sqrt(out_dim), size=(in_dim, out_dim))
    return _base_projection(R.T, name=name)


def get_eye_model(dim=None):
    return EYE_proj(name="Identity", n_components=dim)


def get_CPCA_model(*args, **kwargs):
    return get_PCA_model(*args, **kwargs)


def get_PCA_model(log_dir=None, dim=None, X=None, y1=None, y2=None, y1_name="phone_en_libri", y2_name="spk_libri",
                  whiten=False):
    if log_dir is not None:
        with open(os.path.join(log_dir, f"{y1_name}_pca.pk"), "rb") as in_f:
            y1_pca = pickle.load(in_f)
        with open(os.path.join(log_dir, f"{y2_name}_pca.pk"), "rb") as in_f:
            y2_pca = pickle.load(in_f)
    else:
        if y1 is not None:
            X1, _ = get_mean_agg(X, y1)
            y1_pca = PCA(whiten=whiten).fit(X1)

            if y2 is not None:
                X2, _ = get_mean_agg(X, y2)
                y2_pca = PCA().fit(X2)
                return PCA_proj(y1_pca, name=y1_name, n_components=dim), PCA_proj(y2_pca, name=y2_name, n_components=dim)
            else:
                return PCA_proj(y1_pca, name=y1_name, n_components=dim)
        else:
            y = PCA(whiten=whiten).fit(X)
            return PCA_proj(y, name="PCA", n_components=dim)


def get_LDA_model(log_dir=None, dim=None, X=None, y1=None, y2=None, y1_name="phone", y2_name="spk"):
    if log_dir is not None:
        pass
    else:
        y1_lda = LDA().fit(X, y1)
        if y2 is None:
            return LDA_proj(y1_lda, name=y1_name, n_components=dim)
        else:
            y2_lda = LDA().fit(X, y2)
            return LDA_proj(y1_lda, name=y1_name, n_components=dim), LDA_proj(y2_lda, name=y2_name, n_components=dim)


def get_MLR_model(*argv, **kwargs):
    return get_Linear_model(*argv, **kwargs)


def get_Linear_model(log_dir=None, dim=None, X=None, y1=None, y2=None, y1_name="phone", y2_name="spk",
                     save_path=None, norm=False, fit_intercept=False, subtract_mean=False):
    return_clfs = []
    if subtract_mean:
        X_mean = X.mean(axis=0, keepdims=True)
        X = X - X_mean
    else:
        X_mean = 0

    if log_dir is not None and os.path.exists(os.path.join(log_dir, f"{y1_name}_linear_clf.pk")):
        with open(os.path.join(log_dir, f"{y1_name}_linear_clf.pk"), "rb") as in_f:
            clf_y1 = pickle.load(in_f)
        return_clfs.append(LinearCLF_proj(clf_y1, name=f"Linear_{y1_name}", mean=X_mean, n_components=dim, norm=norm, projection_subtract_mean=subtract_mean))
    else:
        clf_y1 = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=fit_intercept, intercept_scaling=1,
                                    multi_class="auto", solver='lbfgs', max_iter=200, verbose=0, n_jobs=8)
        clf_y1.fit(X, y1)
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{y1_name}_linear_clf.pk"), "wb") as out_f:
                pickle.dump(clf_y1, out_f)
        return_clfs.append(LinearCLF_proj(clf_y1, name=f"Linear_{y1_name}", mean=X_mean, n_components=dim, norm=norm, projection_subtract_mean=subtract_mean))

    if log_dir is not None and os.path.exists(os.path.join(log_dir, f"{y2_name}_linear_clf.pk")):
        with open(os.path.join(log_dir, f"{y2_name}_linear_clf.pk"), "rb") as in_f:
            clf_y2 = pickle.load(in_f)
        return_clfs.append(LinearCLF_proj(clf_y2, name=f"Linear_{y2_name}", mean=X_mean, n_components=dim, norm=norm, projection_subtract_mean=subtract_mean))
    elif y2 is not None:
        clf_y2 = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=fit_intercept, intercept_scaling=1,
                                    multi_class="auto", solver='lbfgs', max_iter=200, verbose=0, n_jobs=8)
        clf_y2.fit(X, y2)
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{y2_name}_linear_clf.pk"), "wb") as out_f:
                pickle.dump(clf_y2, out_f)
        return_clfs.append(LinearCLF_proj(clf_y2, name=f"Linear_{y2_name}", mean=X_mean, n_components=dim, norm=norm, projection_subtract_mean=subtract_mean))

    if len(return_clfs) == 1:
        return return_clfs[0]
    return tuple(return_clfs)
