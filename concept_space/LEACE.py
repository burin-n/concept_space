from dataclasses import dataclass
from torch import Tensor
from concept_erasure import LeaceEraser, LeaceFitter
from concept_erasure.caching import cached_property, invalidates_cache
import torch

# https://github.com/EleutherAI/concept-erasure/blob/main/concept_erasure/leace.py
@dataclass(frozen=True)
class LeaceEraser_v2:
    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None
    sigma_xx: Tensor
    sigma_xz: Tensor
    sigma_zz: Tensor
    W: Tensor
    W_inv: Tensor

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "LeaceEraser_v2":
        """Convenience method to fit a LeaceEraser on data and return it."""        
        return LeaceFitter_v2.fit(x, z, **kwargs).eraser

    def shape(self):
        if self.bias is not None:
            return self.proj_left.shape, self.proj_right.shape, self.bias.shape
        else:
            return self.proj_left.shape, self.proj_right.shape

    def __call__(self, x: Tensor, dim: int = None) -> Tensor:
        """Apply the projection to the input tensor."""
        delta = x - self.bias if self.bias is not None else x
        dim = dim if dim is not None else self.proj_left.shape[1]
        # Ensure we do the matmul in the most efficient order.
        x_ = x - (delta @ self.proj_right.mH)[:, :dim] @ self.proj_left.mH[:dim]
        return x_.type_as(x)

    @property
    def P(self) -> Tensor:
        """The projection matrix."""
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def to(self, device: torch.device | str) -> "LeaceEraser_v2":
        """Move eraser to a new device."""
        return LeaceEraser_v2(
            self.proj_left.to(device),
            self.proj_right.to(device),
            self.bias.to(device) if self.bias is not None else None,
            self.sigma_xx.to(device),
            self.sigma_xz.to(device),
            self.sigma_zz.to(device) if self.sigma_zz is not None else None,
            self.W.to(device),
            self.W_inv.to(device),
        )


class LeaceFitter_v2(LeaceFitter):
    def __init__(self, *args, **kwargs):
        
        if kwargs["method"] == "cca":
            kwargs["method"] = "leace"
            super().__init__(*args, **kwargs)
            self.method = "cca"
            self.sigma_zz_ = torch.zeros(self.z_dim, self.z_dim, device=kwargs["device"], dtype=kwargs["dtype"])
        else:
            super().__init__(*args, **kwargs)
            self.sigma_zz_ = None

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "LeaceFitter":
        """Convenience method to fit a LeaceFitter on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape
        fitter = LeaceFitter_v2(d, k, device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "LeaceFitter":
        """Update the running statistics with a new batch of data."""
        d, c = self.sigma_xz_.shape
        x = x.reshape(-1, d).type_as(self.mean_x)
        n, d2 = x.shape

        assert d == d2, f"Unexpected number of features {d2}"
        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n
        delta_x2 = x - self.mean_x

        # Update the covariance matrix of X if needed (for LEACE)
        if self.method in ["leace", "cca"]:
            assert self.sigma_xx_ is not None
            self.sigma_xx_.addmm_(delta_x.mH, delta_x2)

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(dim=0) / self.n
        delta_z2 = z - self.mean_z

        # Update the cross-covariance matrix
        self.sigma_xz_.addmm_(delta_x.mH, delta_z2)

        if self.method == "cca":
            assert self.sigma_zz_ is not None
            self.sigma_zz_.addmm_(delta_z.mH, delta_z2)

        return self

    @classmethod
    def compute_whitening(cls, sigma):
        L, V = torch.linalg.eigh(sigma)
        # Threshold used by torch.linalg.pinv
        mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)
        # Assuming PSD; account for numerical error
        L.clamp_min_(0.0)            
        W = V * torch.where(mask, L.rsqrt(), 0.0) @ V.mH
        W_inv = V * torch.where(mask, L.sqrt(), 0.0) @ V.mH
        return W, W_inv


    @cached_property
    def eraser(self) -> LeaceEraser_v2:
        """Erasure function lazily computed given the current statistics."""
        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)

        # Compute the whitening and unwhitening matrices.
        if self.method in ["leace", "cca"]:
            W, W_inv = self.compute_whitening(self.sigma_xx)
            if self.method == "cca":
                W_z, W_inv_z = self.compute_whitening(self.sigma_zz_)
            else:
                eye_z = torch.eye(self.z_dim, device=self.mean_z.device, dtype=self.mean_z.dtype)
                W_z, W_inv_z = eye_z, eye_z
        else:
            W, W_inv = eye, eye

        if self.method == "cca":
            u, s, _ = torch.linalg.svd(W @ self.sigma_xz @ W_z, full_matrices=False)
        else:
            u, s, _ = torch.linalg.svd(W @ self.sigma_xz, full_matrices=False)

        # Basis for the column space of sigma_xz; throw away tiny singular values.
        u *= s > self.svd_tol

        proj_left = W_inv @ u
        proj_right = u.mH @ W

        if self.constrain_cov_trace and self.method == "leace":
            P = eye - proj_left @ proj_right

            # Prevent the covariance trace from increasing
            sigma = self.sigma_xx
            old_trace = torch.trace(sigma)
            new_trace = torch.trace(P @ sigma @ P.mH)

            # If applying the projection matrix increases the variance, this might
            # cause instability, especially when erasure is applied multiple times.
            # We regularize toward the orthogonal projection matrix to avoid this.
            if new_trace.real > old_trace.real:
                Q = eye - u @ u.mH

                # Set up the variables for the quadratic equation
                x = new_trace
                y = 2 * torch.trace(P @ sigma @ Q.mH)
                z = torch.trace(Q @ sigma @ Q.mH)
                w = old_trace

                # Solve for the mixture of P and Q that makes the trace equal to the
                # trace of the original covariance matrix
                discr = torch.sqrt(
                    4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y**2
                )
                alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
                alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

                # Choose the positive root
                alpha = torch.where(alpha1.real > 0, alpha1, alpha2).clamp(0, 1)
                P = alpha * P + (1 - alpha) * Q

                # TODO: Avoid using SVD here
                u, s, vh = torch.linalg.svd(eye - P)
                proj_left = u * s.sqrt()
                proj_right = vh * s.sqrt()

        sigma_xx = None if self.method == "orth" else self.sigma_xx
        sigma_zz = None if self.method != "cca" else self.sigma_zz_

        return LeaceEraser_v2(
            proj_left, proj_right, bias=self.mean_x if self.affine else None,
            sigma_xx=sigma_xx, sigma_xz=self.sigma_xz, sigma_zz=sigma_zz,
            W=W,
            W_inv=W_inv
        )

    @property
    def sigma_zz(self) -> Tensor:
        assert self.n > 1, "Call update() before accessing sigma_zz"
        assert (
            self.sigma_zz_ is not None
        ), "Covariance statistics are not being tracked for X"
        # Accumulated numerical error may cause this to be slightly non-symmetric.
        S_hat = (self.sigma_zz_ + self.sigma_zz_.mH) / 2
        return S_hat / (self.n - 1)
