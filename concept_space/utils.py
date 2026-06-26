import numpy as np


# (samples x dim), (samples,)
def get_mean_agg(features, labels):
    feats_agg = []
    labels_uniq = np.array(sorted(np.unique(labels)))
    for label in labels_uniq:
        feats_agg.append(
            features[labels == label, :].mean(axis=0)
        )
    feats_agg = np.array(feats_agg)
    return feats_agg, labels_uniq

# A: (n x dim), B: (n x dim)
def cosine_sim_pairs(A, B):
    A_norm = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=-1, keepdims=True)
    X_dot = np.multiply(A_norm, B_norm).sum(axis=1)
    # (n, 1) 
    return X_dot

def dot_pairs(A, B):
    X_dot = np.multiply(A, B).sum(axis=1)
    # (n, 1) 
    return X_dot

# # A: (n x dim), B: (n x dim)
# def cosine_sim_all_pairs(A, B):
#     A_norm = A / np.linalg.norm(A, axis=-1, keepdims=True)
#     B_norm = B / np.linalg.norm(B, axis=-1, keepdims=True)
#     X_dot = np.dot(A_norm, B_norm.T)
#     # (n, n) 
#     return X_dot

# A: (dim), B: (dim) or A: (n, dim), B: (n, dim)
def cosine_sim(A, B):
    A_norm = np.linalg.norm(A, axis=-1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=-1, keepdims=True)
    # print((A/A_norm).shape, (B/B_norm).shape)
    X_dot = np.dot(A/A_norm, (B/B_norm).T)
    # (n x n ) or scalar
    # print(f"A has the average norm of {A_norm.mean()}")
    # print(f"B has the average norm of {B_norm.mean()}")
    return X_dot


def angle(A, B):
    cos_sim = cosine_sim(A, B)
    angles = np.rad2deg(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    return angles

def get_sampling_rate_from_processor(processor):
    expected_sampling = getattr(processor, "sampling_rate", None)
    if expected_sampling is None:
        try:
            expected_sampling = getattr(processor.feature_extractor, "sampling_rate", None)
        except Exception:
            raise ValueError("Cannot retrieve expected sampling rate from processor")
    if expected_sampling is None:
        raise ValueError("Cannot retrieve expected sampling rate from processor")
    return expected_sampling
