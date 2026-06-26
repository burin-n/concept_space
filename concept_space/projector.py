import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class _base_projection(ABC):

    def __init__(self, components, mean=0, name="projector", projection_method="SVD",
                 providedSVDProj=None, projection_subtract_mean=True):
        self.components = components
        self.mean = mean
        self.name = name
        self.n_components = len(components)
        self.projection_subtract_mean = projection_subtract_mean

        assert projection_method in ["SVD", "LeastSQ", "providedSVD"]
        self.projection_method = projection_method
        if projection_method == "providedSVD":
            assert providedSVDProj is not None
            self.P_U, self.P_Sigma, self.P_vh = providedSVDProj
            self.projection_matrix_rank = len(self.P_Sigma)

    def transform(self, X, n_components=None):
        n_components = self.n_components if n_components is None else n_components
        transform_matrix = getattr(self, "transform_matrix", None)
        transform_matrix_rank = getattr(self, "transform_matrix_rank", 0)
        if transform_matrix is None or transform_matrix_rank != n_components:
            U, S, Vh = np.linalg.svd(self.components, full_matrices=False)
            Vh = Vh[:n_components]
            S = S[:n_components].reshape(-1, 1)
            U = U[:n_components, :n_components]
            transform_matrix = U @ (S * Vh)
            self.transform_matrix = transform_matrix
            self.transform_matrix_rank = n_components
        return (X - self.mean) @ self.transform_matrix.T

    def transformation(self, X):
        # x: (batch_size, in_dim)
        # A: (out_dim, in_dim)
        row_bases = getattr(self, "row_bases", None)
        col_bases = getattr(self, "col_bases", None)

        if col_bases is None:
            # (out_dim, rank), (rank), (rank, in_dim)
            U, S, Vh = np.linalg.svd(self.components, full_matrices=False)
            # (in_dim, rank)
            col_bases = U
            row_bases = Vh.T
            self.col_bases = col_bases
            self.row_bases = row_bases

        # (batch size, rank)
        X_ = X @ row_bases
        bias = getattr(self, "bias", None)
        if bias is not None and bias != False:
            raise NotImplementedError()
        # (batch size, in_dim)
        return X_ @ row_bases.T

    def projection(self, X, n_components=None, subtract_mean=None):
        # X shape (sample, in_dim)
        if n_components is None:
            n_components = self.n_components
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean

        if subtract_mean:
            X = X - self.mean

        projection_matrix = getattr(self, "projection_matrix", None)
        if projection_matrix is None or self.projection_matrix_rank != n_components:
            components = self.components.T
            if self.projection_method == "LeastSQ":
                self.projection_matrix = components @ np.linalg.pinv(components.T @ components) @ components.T
            elif self.projection_method == "SVD":
                U, S, Vh = np.linalg.svd(self.components, full_matrices=False)
                V = Vh.T[:, :n_components]
                projection_matrix = V @ V.T
                self.projection_matrix = projection_matrix
                self.projection_matrix_rank = n_components
            elif self.projection_method == "providedSVD":
                projection_matrix = self.P_U[:, :n_components] * self.P_Sigma[:n_components] @ self.P_vh[:n_components]
                self.projection_matrix_rank = n_components
                self.projection_matrix = projection_matrix

        return X @ self.projection_matrix

    def rejection(self, X, method="projection", n_components=None, subtract_mean=None):
        # X shape (sample, in_dim)
        if method == "projection":
            return X - self.projection(X, n_components=n_components, subtract_mean=subtract_mean)
        elif method == "transformation":
            return X - self.transformation(X)
        elif method == "nullspace":
            return self.nullspace_projection(X)
        else:
            raise NotImplementedError("rejection method: {} is not implemented".format(method))


class PCA_proj(_base_projection):
    def __init__(self, pca_model, name="projector", n_components=None):
        self.pca_model = pca_model
        if n_components is None:
            self.n_components = pca_model.n_components
        else:
            self.n_components = n_components
        # (out_dim, in_dim)
        super().__init__(
            components=pca_model.components_[:self.n_components],
            mean=pca_model.mean_,
            name=name
        )

    def transform(self, X, n_components=None):
        if n_components is None:
            n_components = self.n_components
        return self.pca_model.transform(X)[:, :n_components]

    def projection(self, X, n_components=None, subtract_mean=None):
        if n_components is None:
            n_components = self.n_components
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean

        if subtract_mean:
            X = X - self.mean

        projection_matrix = getattr(self, "projection_matrix", None)
        if projection_matrix is None or self.projection_matrix_rank != n_components:
            projection_matrix = self.pca_model.components_[:n_components].T @ self.pca_model.components_[:n_components]
            self.projection_matrix_rank = n_components
        return X @ projection_matrix

class LDA_proj(_base_projection):
    def __init__(self, lda_model, name="projector", n_components=None):
        self.lda_model = lda_model
        if n_components is None:
            self.n_components = lda_model.n_components
        else:
            self.n_components = n_components
        super().__init__(
            components=lda_model.scalings_.T[:self.n_components],
            mean=lda_model.xbar_,
            name=name
        )

    def transform(self, X, n_components=None):
        if n_components is None:
            n_components = self.n_components
        # (samples, n_comp)
        return self.lda_model.transform(X)[:, :self.n_components]


class LinearCLF_proj(_base_projection):
    def __init__(self, linear_model, name="LinearCLF", mean=0, n_components=None, norm=False, projection_method="SVD", projection_subtract_mean=True):
        self.linear_model = linear_model
        if n_components is None:
            self.n_components = linear_model.coef_.shape[0]
        else:
            self.n_components = n_components
        components = linear_model.coef_[:self.n_components]
        if norm:
            components = components / np.linalg.norm(components, axis=-1, keepdims=True)
        super().__init__(
            components=components,
            mean=mean,
            name=name,
            projection_method=projection_method,
            projection_subtract_mean=projection_subtract_mean
        )


class LEACE_proj(_base_projection):
    def __init__(self, model_LEACE, name="LEACE", n_components=None):
        self.model = model_LEACE
        self.mean = model_LEACE.bias.numpy() if model_LEACE.bias is not None else 0
        self.n_components = n_components if n_components is not None else model_LEACE.proj_left.shape[-1]
        self.proj_right = np.asarray(model_LEACE.proj_right.mH)
        self.proj_left = np.asarray(model_LEACE.proj_left.mH)
        self.proj_all = self.proj_right @ self.proj_left
        self.components = self.proj_right.T
        self.name = name

    def projection(self, X, n_components=None, subtract_mean=True):
        """Apply the projection to the input tensor."""
        if subtract_mean:
            delta = X - self.mean if self.mean is not None else X
        else:
            delta = X
        if n_components is None:
            n_components = self.n_components
        X_ = delta @ self.proj_right[:, :n_components] @ self.proj_left[:n_components]
        return X_

    def transform(self, X, n_components=None):
        delta = X - self.mean if self.mean is not None else X
        if n_components is None:
            n_components = self.n_components
        X_ = delta @ self.proj_right[:, :n_components]
        return X_

    def rejection(self, X, n_components=None, subtract_mean=True):
        return X - self.projection(X, n_components=n_components, subtract_mean=subtract_mean)


class EYE_proj(_base_projection):
    def __init__(self, name="EYE", n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.name = name

    def transform(self, X):
        # (samples, n_comp)
        return X

    def projection(self, X):
        return X

    def rejection(self, X, method=None):
        return X - X
