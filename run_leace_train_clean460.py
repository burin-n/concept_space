import argparse
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LEACE spaces on LibriSpeech train-clean-460 and evaluate on test-clean."
    )
    parser.add_argument("score_dir", help="Directory for evaluation score CSVs")
    parser.add_argument("--layer", type=int, default=11, help="HuBERT layer to train/evaluate")
    parser.add_argument("--model", default="facebook/hubert-base-ls960", help="HuggingFace model name")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Utterances per train chunk")
    parser.add_argument("--space-dir", default="save_space", help="Directory for saved LEACE spaces")
    parser.add_argument("--train-name", default="train460", help="Prefix used in saved spaces and score CSVs")
    parser.add_argument(
        "--spk-components",
        nargs="+",
        type=int,
        default=[40, 100, 200, 300, 400, 500, 600, 700, 768],
        help="Speaker LEACE component counts to evaluate",
    )
    parser.add_argument(
        "--classifier-type",
        choices=["linear", "non-linear"],
        default="linear",
        help="Classifier used for evaluation",
    )
    parser.add_argument(
        "--overwrite-space",
        action="store_true",
        help="Retrain and overwrite saved LEACE spaces even if pickles already exist",
    )
    train_cache_group = parser.add_mutually_exclusive_group()
    train_cache_group.add_argument(
        "--cache-train",
        dest="cache_train",
        action="store_true",
        help="Read/write cached train-clean-100/360 features instead of the default in-memory train inference",
    )
    train_cache_group.add_argument(
        "--no-cache-train",
        dest="cache_train",
        action="store_false",
        help="Compute train-clean-100/360 features in memory without reading or writing train feature caches (default)",
    )
    parser.set_defaults(cache_train=False)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    return args

def space_paths(args, name):
    layer_dir = os.path.join(args.space_dir, name, str(args.layer))
    return (
        os.path.join(layer_dir, f"leace-{args.train_name}-phone.pk"),
        os.path.join(layer_dir, f"leace-{args.train_name}-spk.pk"),
    )


def load_train_resources(args, processor, data, load_dataset, Audio):
    alignment_dir = os.environ.get("ALIGNMENT_DIR", "")
    if not alignment_dir:
        raise ValueError("ALIGNMENT_DIR must point to the forced-alignment directory.")

    resources, phone_labels, speaker_ids = [], [], []
    for split_id in ["100", "360"]:
        hf_split = f"train.{split_id}"
        cache_name = f"train.clean.{split_id}"
        alignment_path = os.path.join(alignment_dir, f"train-clean-{split_id}.ali")
        if not os.path.exists(alignment_path):
            raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

        print(f"Loading dataset clean/{hf_split}")
        ds = load_dataset("openslr/librispeech_asr", name="clean", split=hf_split)
        ds = ds.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        ali = data.get_alignment(alignment_path, cd_phone_sp=None)

        resources.append((hf_split, cache_name, ds, ali))
        phone_labels.extend(ali["phone"].tolist())
        speaker_ids.extend(ds["speaker_id"])

    phone2id, _ = data.generate_class_index(phone_labels)
    for idx, (hf_split, cache_name, ds, ali) in enumerate(resources):
        ali = ali.copy()
        ali["phone_id"] = ali["phone"].map(phone2id)
        resources[idx] = (hf_split, cache_name, ds, ali)

    return resources, len(phone2id), speaker_ids


def train_spaces(args, model, processor, cache_root, name):
    import numpy as np
    import torch
    from datasets import Audio, load_dataset
    from einops import rearrange
    from sklearn.preprocessing import OneHotEncoder
    from tqdm import tqdm
    from concept_space import data
    from concept_space.LEACE import LeaceFitter_v2
    from concept_space.helper.inference_libri import inference, retrive_logit_at_layer
    from concept_space.projector import LEACE_proj

    def infer_chunk(chunk):
        id2logit = {}
        for sample in tqdm(chunk, desc=f"Inferring"):
            inputs = processor(sample["audio"]["array"], sampling_rate=processor.sampling_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = model(inputs.input_values.to(model.device), output_hidden_states=True)
            id2logit[sample["id"]] = rearrange(
                outputs.hidden_states[args.layer].to("cpu").float().numpy(),
                "() t h -> t h",
            )
        return id2logit

    resources, n_phone, speakers = load_train_resources(args, processor, data, load_dataset, Audio)
    spk_encoder = OneHotEncoder(sparse_output=False).fit(np.asarray(speakers).reshape(-1, 1))
    dim = getattr(model.config, "hidden_size", None)
    if dim is None:
        raise ValueError("Could not infer model hidden size from model.config.hidden_size")

    fit_phone = LeaceFitter_v2(dim, n_phone, dtype=torch.float, svd_tol=1e-7, method="leace")
    fit_spk = LeaceFitter_v2(dim, len(spk_encoder.categories_[0]), dtype=torch.float, svd_tol=1e-7, method="leace")

    for hf_split, cache_name, ds, ali in resources:
        split_cache = os.path.join(cache_root, cache_name)
        for start in range(0, len(ds), args.chunk_size):
            end = min(len(ds), start + args.chunk_size)
            print(f"{hf_split}: processing utterances {start}:{end}")
            chunk = ds.select(range(start, end))
            utt_ids = chunk["id"]

            if not args.cache_train:
                id2logit = infer_chunk(chunk)
            else:
                try:
                    print(f"Searching for cache for layer {args.layer}")
                    logits = retrive_logit_at_layer(utt_ids, args.layer, split_cache)
                    print(f"Found cache for layer {args.layer}")
                except Exception:
                    print(f"Cache not found, running inference for layer {args.layer}")
                    inference(chunk, model, processor, cache_dir=split_cache, return_hidden=False, layer_id=args.layer)
                    logits = retrive_logit_at_layer(utt_ids, args.layer, split_cache)
                id2logit = dict(zip(utt_ids, logits))

            X, y_phone, y_spk = data.load_data(chunk, id2logit, ali, njobs=data.DATA_N_JOBS)
            X = torch.from_numpy(X).float()
            fit_phone.update(X, torch.nn.functional.one_hot(torch.from_numpy(y_phone).long(), num_classes=n_phone))
            fit_spk.update(X, torch.from_numpy(spk_encoder.transform(y_spk.reshape(-1, 1))))

    phone_proj = LEACE_proj(fit_phone.eraser, "LEACE_phone_clean")
    spk_proj = LEACE_proj(fit_spk.eraser, "LEACE_spk_clean")
    phone_path, spk_path = space_paths(args, name)
    os.makedirs(os.path.dirname(phone_path), exist_ok=True)
    with open(phone_path, "wb") as out_f:
        pickle.dump(phone_proj, out_f)
    with open(spk_path, "wb") as out_f:
        pickle.dump(spk_proj, out_f)
    print(f"Saved {phone_path}")
    print(f"Saved {spk_path}")
    return phone_proj, spk_proj


def load_or_train_spaces(args, model, processor, cache_root, name):
    phone_path, spk_path = space_paths(args, name)
    if not args.overwrite_space and os.path.exists(phone_path) and os.path.exists(spk_path):
        print(f"Loading saved LEACE spaces from {os.path.dirname(phone_path)}")
        with open(phone_path, "rb") as in_f:
            phone_proj = pickle.load(in_f)
        with open(spk_path, "rb") as in_f:
            spk_proj = pickle.load(in_f)
        return phone_proj, spk_proj
    return train_spaces(args, model, processor, cache_root, name)


def evaluate_spaces(args, model, processor, cache_root, phone_proj, spk_proj):
    from concept_space import data, evaluate
    from concept_space.helper.helper import get_eye_model
    from concept_space.projector import LEACE_proj

    X_dev, y_phone_dev, y_spk_dev, X_test, y_phone_test, y_spk_test, _, _ = data.make_classification_data(
        model,
        processor,
        args.layer,
        cache_root,
        alignment_ground_truth_shift=0,
        alignment_context_window=[0, 0],
        data_include_dev_other=False,
        data_train_test_spk_overlap=True,
        norm_spk_index=False,
        split_ratio=0.3,
        split="test",
    )
    data_clean_set = (X_dev, y_phone_dev, y_spk_dev, X_test, y_phone_test, y_spk_test)

    projections = [get_eye_model(), phone_proj]
    names = ["EYE_clean", "LEACE_phone_clean"]
    max_components = spk_proj.model.proj_left.shape[-1]
    for n in args.spk_components:
        if n > max_components:
            raise ValueError(f"Requested {n} speaker components, but only {max_components} are available.")
        projections.append(LEACE_proj(spk_proj.model, f"LEACE-comp{n}_spk_clean", n_components=n))
        names.append(f"LEACE-comp{n}_spk_clean")

    for projection, name in zip(projections, names):
        print(f"Evaluating {name}")
        evaluate.eval_v3_wrapper(
            projection,
            name,
            args.train_name,
            data_clean_set,
            data_other_set=None,
            save_folder=args.score_dir,
            suffix_y1="-phone",
            suffix_y2="-spk",
            classifier_type=args.classifier_type,
        )


def main():
    args = parse_args()

    from dotenv import load_dotenv
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    load_dotenv()
    processor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    print(f"Loading model {args.model}")
    model = AutoModel.from_pretrained(args.model).to("cpu")
    model.eval()

    name = args.model.split("/")[-1]
    cache_root = os.path.join(os.environ.get("CACHE_DIR", "cache"), name, "Librispeech")
    phone_proj, spk_proj = load_or_train_spaces(args, model, processor, cache_root, name)
    evaluate_spaces(args, model, processor, cache_root, phone_proj, spk_proj)


if __name__ == "__main__":
    main()
