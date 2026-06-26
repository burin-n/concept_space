import argparse
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from concept_space import evaluate
from concept_space import data
from concept_space.helper.helper import (
    get_LEACE_model, get_Linear_model, get_random_model,
    get_PCA_model, get_eye_model, get_LDA_model,
)


def run_test_suite(processor, model, layer_id, cache_dir, log_dir):
    X_train_clean, y_phone_train_clean, y_spk_train_clean = data.make_classification_data(
        model, processor, layer_id, cache_dir,
        alignment_ground_truth_shift=0,
        alignment_context_window=[0, 0],
        data_include_dev_other=False,
        data_train_test_spk_overlap=False,
        norm_spk_index=False,
        split_ratio=None)

    X_dev_clean, y_phone_dev_clean, y_spk_dev_clean, X_test_clean, y_phone_test_clean, y_spk_test_clean, _, _ = data.make_classification_data(
        model, processor, layer_id, cache_dir,
        alignment_ground_truth_shift=0,
        alignment_context_window=[0, 0],
        data_include_dev_other=False,
        data_train_test_spk_overlap=True,
        norm_spk_index=False,
        split_ratio=0.3,
        split="test")

    train_data = torch.from_numpy(np.concatenate([X_train_clean], axis=0))
    train_phone_data = np.concatenate([y_phone_train_clean], axis=0)
    train_spk_data = np.concatenate([y_spk_train_clean], axis=0)
    LEACE_phone_clean, LEACE_spk_clean = get_LEACE_model(train_data, train_phone_data, train_spk_data, leace_use_whitening=True)

    train_data = torch.from_numpy(np.concatenate([X_train_clean], axis=0))
    train_phone_data = np.concatenate([y_phone_train_clean], axis=0)
    train_spk_data = np.concatenate([y_spk_train_clean], axis=0)
    COV_phone_clean, COV_spk_clean = get_LEACE_model(train_data, train_phone_data, train_spk_data,
                                                      model_y1_suffix="COV_phone", model_y2_suffix="COV_spk",
                                                      method="orth")

    train_data = np.concatenate([X_train_clean], axis=0)
    train_phone_data = np.concatenate([y_phone_train_clean], axis=0)
    train_spk_data = np.concatenate([y_spk_train_clean], axis=0)
    Linear_phone_clean, Linear_spk_clean = get_Linear_model(
        X=train_data, y1=train_phone_data, y2=train_spk_data,
        y1_name="phone", y2_name="spk",
        save_path="save_linear_model_layer4/train-clean",
        norm=True)

    RANDOM_phone_clean = [get_random_model(in_dim=768, out_dim=39) for _ in range(5)]
    RANDOM_spk_clean = [get_random_model(in_dim=768, out_dim=len(np.unique(train_spk_data))) for _ in range(5)]

    PCA_phone_clean, PCA_spk_clean = get_PCA_model(
        X=train_data, y1=train_phone_data, y2=train_spk_data,
        y1_name="phone", y2_name="spk")
    EYE_model = get_eye_model()

    LDA_phone_clean, LDA_spk_clean = get_LDA_model(
        X=train_data, y1=train_phone_data, y2=train_spk_data,
        y1_name="phone", y2_name="spk")

    data_clean_set = (X_dev_clean, y_phone_dev_clean, y_spk_dev_clean, X_test_clean, y_phone_test_clean, y_spk_test_clean)
    data_other_set = None

    save_folder = f"{log_dir}/layer_{layer_id}"
    proj_train_data = "dev-clean"
    for projection_to_use, proj_name in zip(
        [EYE_model, *RANDOM_phone_clean, *RANDOM_spk_clean, Linear_phone_clean, Linear_spk_clean,
         LDA_phone_clean, LDA_spk_clean, PCA_phone_clean, PCA_spk_clean,
         COV_phone_clean, COV_spk_clean, LEACE_phone_clean, LEACE_spk_clean],
        ["EYE_clean",
         *[f"RANDOM{i}_phone_clean" for i in range(5)],
         *[f"RANDOM{i}_spk_clean" for i in range(5)],
         "LC_phone_clean", "LC_spk_clean", "LDA_phone_clean", "LDA_spk_clean",
         "PCA_phone_clean", "PCA_spk_clean", "COV_phone_clean", "COV_spk_clean",
         "LEACE_phone_clean", "LEACE_spk_clean"]):
        print("Evaluating", proj_name)
        evaluate.eval_v3_wrapper(projection_to_use, proj_name, proj_train_data, data_clean_set, data_other_set,
                                  save_folder=save_folder, suffix_y1="-phone", suffix_y2="-spk",
                                  classifier_type="linear")


def main():
    parser = argparse.ArgumentParser(description="ACL concept-space test suite")
    parser.add_argument("log_dir", help="Output directory for evaluation scores")
    parser.add_argument("--layers", nargs="+", type=int, default=[11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
                        help="Layer IDs to evaluate")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    processor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True)

    model_uri_name = "facebook/hubert-base-ls960"
    print("running for", model_uri_name)
    model = AutoModel.from_pretrained(model_uri_name)
    model_name = model_uri_name.split("/")[-1]
    model = model.to("cpu")
    model.eval()

    cache_dir = os.path.join(os.environ.get("CACHE_DIR", "cache"), model_name, "Librispeech")
    for layer_id in args.layers:
        run_test_suite(processor, model, layer_id, cache_dir, args.log_dir)


if __name__ == "__main__":
    main()
