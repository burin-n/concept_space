import numpy as np
from datasets import load_dataset, Audio
import os
from concept_space.helper.inference_libri import inference, retrive_logit_at_layer, inference_melspectrogram
import pandas as pd
import re
from concept_space import utils
import logging
import time
import pickle
import math
from joblib import Parallel, delayed


DATA_N_JOBS = int(os.environ.get("DATA_N_JOBS", 4))
SPK_TEST_SPLIT_DIR = os.environ.get("SPK_TEST_SPLIT_DIR", "save_spk_test_split")

model_to_subsampling_factor_ = {
    "hubert": 320,
}

model_to_sampling_rate_ = {
    "hubert": 16000,
}


def model_to_sampling_rate(model_name):
    model_name = model_name.lower()
    if model_name in model_to_sampling_rate_:
        return model_to_sampling_rate_[model_name]
    else:
        raise ValueError(f"Unknown model type for sampling rate: {model_name}")


def model_to_subsampling_factor(model_name, layer_id=None):
    model_name = model_name.lower()
    if model_name in model_to_subsampling_factor_:
        subsampling_factor = model_to_subsampling_factor_[model_name]
        if type(subsampling_factor) == dict:
            if layer_id is not None and str(layer_id) in subsampling_factor.keys():
                return subsampling_factor[str(layer_id)]
            return subsampling_factor['default']
        else:
            return subsampling_factor
    else:
        raise ValueError(f"Unknown model type for subsampling factor: {model_name}")


def clean(txt):
    txt = txt.split("_")[0]
    return re.sub("[0-9]", "", txt)


def time_to_idx(t, sr=16000, subsampling_factor=320, rounding="floor"):
    if rounding == "floor":
        idx = int(math.floor(t * sr / subsampling_factor))
    if rounding == "ceiling":
        idx = int(math.ceil(t * sr / subsampling_factor))
    return idx


def idx_to_time(idx, sr=16000, subsampling_factor=320):
    return idx * subsampling_factor / sr


def is_sil(phone, cd_phone_sep="_", ignore_phones_extra=[]):
    ignore_phones = ["<eps>", "<sil>", "<SIL>"] + ignore_phones_extra
    n_sil = 0
    for p_ in phone.split(cd_phone_sep):
        if p_ in ignore_phones:
            n_sil += 1
    return n_sil > 0


def get_context_phone(alignment, cd_phone_sp="_", context_window=[0, 0]):
    if cd_phone_sp is None:
        return alignment["phone"]
    ct_phone = []
    left_context, right_context = context_window

    if sum(context_window) == 0:
        return alignment["phone"]

    for utt_id in alignment["utt_id"].unique():
        alignment_subset = alignment[alignment["utt_id"] == utt_id]
        for i in range(len(alignment_subset["phone"])):
            ct_phone_ = alignment_subset["phone"].iloc[i - left_context: i + right_context + 1]
            ct_phone.append(cd_phone_sp.join(ct_phone_))

    return ct_phone


def generate_class_index(classes):
    class_uniq = np.unique(classes)
    class2id = {c_: i_ for i_, c_ in enumerate(sorted(class_uniq))}
    id2class = list(class2id)
    return class2id, id2class


def get_alignment(alignment_file, context_window=[0, 0], cd_phone_sp="_", fn_clean_phone=clean,
                  sampling_rate=16000, subsampling_factor=320, drop_silence=True):
    # context_window; left context size, right context size
    # context_window = [0, 0] is monophone
    # context_window = [1, 0] is biphone for past-phone_cur-phone
    # context_window = [0, 1] is biphone with cur-phone_future-phone
    # context_window = [1, 1] is triphone
    if type(context_window) == str:
        for potential_sep in [";", ","]:
            try:
                context_window = [int(x) for x in context_window.split(potential_sep)]
                if len(context_window) == 2:
                    break
            except Exception:
                pass
    if len(context_window) != 2:
        raise ValueError(f"length of context must be 2: {context_window}")

    alignment = pd.read_csv(alignment_file, sep=" ")
    alignment["utt_id"] = alignment["utt_id"].map(lambda x: x.strip("lbi-"))

    alignment["phone"] = alignment["phone"].map(fn_clean_phone)
    ct_phone = get_context_phone(alignment, context_window=context_window, cd_phone_sp=cd_phone_sp)
    alignment["phone"] = ct_phone

    context_window_size = 1 + sum(context_window)
    alignment = alignment[alignment["phone"].map(lambda x: len(x.split(cd_phone_sp)) == context_window_size)]
    if drop_silence:
        alignment = alignment[~alignment["phone"].map(lambda x: is_sil(x, ignore_phones_extra=["SIL", "SPN"]))]

    phone2id, id2phone = generate_class_index(alignment["phone"])

    alignment["phone_id"] = alignment["phone"].map(phone2id)
    alignment["start_time"] = alignment["start_time"].map(lambda t: time_to_idx(t, sr=sampling_rate, subsampling_factor=subsampling_factor))
    alignment["end_time"] = alignment["start_time"] + alignment["phone_dur"].map(lambda t: time_to_idx(t, sr=sampling_rate, subsampling_factor=subsampling_factor, rounding="ceiling"))
    return alignment


def shift_alignment(alignment, shift=0):
    ct_phone = []

    if shift == 0:
        return alignment

    for utt_id in alignment["utt_id"].unique():
        alignment_subset = alignment[alignment["utt_id"] == utt_id]
        for i in range(len(alignment_subset["phone"])):
            if i + shift >= 0 and i + shift < len(alignment_subset):
                ct_phone_ = alignment_subset["phone"].iloc[i + shift]
            else:
                ct_phone_ = ""
            ct_phone.append(ct_phone_)

    alignment_out = alignment.copy()
    alignment_out["phone"] = ct_phone
    alignment_out = alignment_out[alignment_out["phone"].map(lambda x: x != "")]
    return alignment_out


def get_label(utt_id, tot_frame, alignment, EMPTY_LABEL=-1):
    sel_align = alignment[alignment["utt_id"] == utt_id]
    label = np.array([EMPTY_LABEL for _ in range(tot_frame)], dtype=int)
    for target_choice in ["phone_id", "word_id", "syllable_id"]:
        if target_choice in alignment:
            target = target_choice
    for s, t, p in zip(sel_align["start_time"], sel_align["end_time"], sel_align[target]):
        label[s:t] = p
    return label


def process_input(utt_id, id2logit, alignment):
    EMPTY_LABEL = -1
    logit = id2logit[utt_id]
    tot_frame = len(logit)
    label = get_label(utt_id, tot_frame, alignment, EMPTY_LABEL)
    assert logit.shape[0] == label.shape[0]
    logit = logit[label != EMPTY_LABEL]
    label = label[label != EMPTY_LABEL]
    assert logit.shape[0] == label.shape[0]
    return logit, label


def get_data(idx, dataset, id2logit, alignment):
    _dataset = dataset[idx]
    _dataset["logit"], _dataset["label"] = process_input(_dataset["id"], id2logit, alignment=alignment)
    return _dataset


def load_data(dataset, id2logit, alignment, njobs=DATA_N_JOBS, logger=logging, preserve_sequence=False):
    start_time = time.time()
    assert logger is not None

    pre_indices = np.arange(len(dataset))
    step_size = len(dataset) // njobs

    indices = [pre_indices[i * step_size: (i + 1) * step_size] for i in range(njobs)]
    if len(dataset) > njobs * step_size:
        indices[-1] = np.concatenate((indices[-1], pre_indices[njobs * step_size:]))

    indices = [indice_ for indice_ in indices if len(indice_) > 0]
    assert sum([len(x) for x in indices]) == len(dataset)

    logger.info("creating dataset mp with Joblib")

    def load_data_mp(indices_batch, dataset, id2logit, alignment):
        X, y_phone, y_spk = [], [], []
        for _idx in indices_batch:
            _data = get_data(int(_idx), dataset, id2logit, alignment)
            X.append(_data["logit"])
            y_spk.append(np.array([_data["speaker_id"]] * len(_data["logit"]), dtype=int))
            y_phone.append(_data["label"])

        if not preserve_sequence:
            X = np.concatenate(X)
            y_spk = np.concatenate(y_spk)
            y_phone = np.concatenate(y_phone)

        return X, y_phone, y_spk

    results = Parallel(n_jobs=njobs, verbose=0)(
        delayed(load_data_mp)(
            batch_indices, 
            dataset, 
            id2logit, 
            alignment
        ) for batch_indices in indices
    )

    if preserve_sequence:
        X = [utt for res in results for utt in res[0]]
        y_phone = [utt for res in results for utt in res[1]]
        y_spk = [utt for res in results for utt in res[2]]
    else:
        X = np.concatenate([r[0] for r in results])
        y_phone = np.concatenate([r[1] for r in results])
        y_spk = np.concatenate([r[2] for r in results])

    del results
    logger.info("dataset creation done.\n")
    logger.info("dataset creation time: {:.2f}".format(time.time() - start_time))
    return X, y_phone, y_spk


def train_test_split(X, y_phone, y_spk, split_ratio=0.2, leave_spk_out=True, split_index_file="data/train_test_split.npy"):
    if leave_spk_out:
        n_spk_val = int(len(np.unique(y_spk)) * split_ratio)
        spk_val = set()
        for i, spk in enumerate(reversed(y_spk)):
            spk_val.add(spk)
            if len(spk_val) > n_spk_val:
                spk_val.remove(spk)
                break

        X_train, X_test = X[:-i], X[-i:]
        y_phone_train, y_phone_test = y_phone[:-i], y_phone[-i:]
        y_spk_train, y_spk_test = y_spk[:-i], y_spk[-i:]
        assert len(X_train) + len(X_test) == len(X)
        assert len(np.unique(y_spk_train)) + len(np.unique(y_spk_test)) == len(np.unique(y_spk))
        return X_train, y_phone_train, y_spk_train, X_test, y_phone_test, y_spk_test
    else:
        train_size = int((1 - split_ratio) * len(X))
        if split_index_file is not None and os.path.exists(split_index_file):
            choices = np.load(split_index_file)
            print(f"split_index_file found, using: {split_index_file}")
        else:
            print(f"split_index_file not found, creating a new one: {split_index_file}")
            while True:
                choices = np.random.choice(np.arange(len(X)), size=len(X), replace=False)
                y_spk_train, y_spk_test = y_spk[choices[:train_size]], y_spk[choices[train_size:]]
                if set(y_spk_test) - set(y_spk_train) == set():
                    break
            os.makedirs(os.path.dirname(split_index_file) or ".", exist_ok=True)
            np.save(split_index_file, choices)

        y_spk_train, y_spk_test = y_spk[choices[:train_size]], y_spk[choices[train_size:]]
        X_train, X_test = X[choices[:train_size]], X[choices[train_size:]]
        y_phone_train, y_phone_test = y_phone[choices[:train_size]], y_phone[choices[train_size:]]

        assert len(X_train) + len(X_test) == len(X)
        assert len(set(y_spk_train).union(set(y_spk_test))) == len(np.unique(y_spk))
        return X_train, y_phone_train, y_spk_train, X_test, y_phone_test, y_spk_test


def retrive_librispeech_data(model, processor, cache_dir, layer_id=11, subset="clean", split="dev",
                             alignment_context_window=[0, 0], alignment_ground_truth_shift=0,
                             alignment_overwrite=None, preserve_sequence=False,
                             return_format="classification",
                             inference_fallback_function=inference):

    if split == "dev":
        split_ = "validation"
    else:
        split_ = split

    if processor is not None:
        expected_sampling = utils.get_sampling_rate_from_processor(processor)
    else:
        expected_sampling = 16000

    dataset = load_dataset("openslr/librispeech_asr", name=subset, split=split_)
    print("casting sampling rate to", expected_sampling)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=expected_sampling))

    if alignment_overwrite is not None:
        alignment = alignment_overwrite
    else:
        alignment_dir = os.environ.get("ALIGNMENT_DIR", "")
        if not alignment_dir:
            raise ValueError(
                "ALIGNMENT_DIR environment variable must be set to the directory containing "
                "forced-alignment .ali files (e.g. export ALIGNMENT_DIR=/path/to/forced_alignment)."
            )
        alignment_path = os.path.join(alignment_dir, f"{split}-{subset}.ali")
        if not os.path.exists(alignment_path):
            raise ValueError(
                f"Alignment file not found: {alignment_path}. "
                "Check ALIGNMENT_DIR and ensure the file exists."
            )
        alignment = get_alignment(alignment_path, context_window=alignment_context_window)
        alignment = shift_alignment(alignment, alignment_ground_truth_shift)

    cache_dir = os.path.join(cache_dir, f"{split}-{subset}")

    if model is None:
        id2logit = inference_melspectrogram(dataset)
    else:
        try:
            print(f"Searching for cache for layer {layer_id}")
            _logit = retrive_logit_at_layer(dataset["id"], layer_id, cache_dir)
            id2logit = {dataset["id"][i]: _logit[i] for i in range(len(dataset["id"]))}
            del _logit
            print(f"Found cache for layer {layer_id}")
        except Exception:
            print(f"Cache not found, running inference for layer {layer_id}")
            inference_fallback_function(dataset, model, processor, cache_dir=cache_dir, return_hidden=False, layer_id=layer_id)
            _logit = retrive_logit_at_layer(dataset["id"], layer_id, cache_dir)
            id2logit = {dataset["id"][i]: _logit[i] for i in range(len(dataset["id"]))}
            del _logit

    if type(return_format) == str and return_format == "classification":
        X, y_phone, y_spk = load_data(dataset, id2logit, alignment, njobs=DATA_N_JOBS, preserve_sequence=preserve_sequence)
        return X, y_phone, y_spk
    elif type(return_format) == list:
        return_dict = dict()

        if "features" in return_format:
            if "y_phone" in return_format or "y_spk" in return_format:
                X, y_phone, y_spk = load_data(dataset, id2logit, alignment, njobs=DATA_N_JOBS, preserve_sequence=preserve_sequence)
                return_dict["features"] = X
                return_dict["y_phone"] = y_phone
                return_dict["y_spk"] = y_spk
            else:
                X = []
                for utt_id in dataset["id"]:
                    X.append(id2logit[utt_id])
                return_dict["features"] = X

        if "id" in return_format:
            return_dict["id"] = dataset["id"]

        if len(return_format) == 1:
            return return_dict[return_format[0]]
        else:
            return tuple([return_dict[key] for key in return_format])


def normalize_spk_index(y_spk, spkix2spk=None, spk2spkix=None):
    if spkix2spk is None and spk2spkix is None:
        spkix2spk = np.unique(y_spk)
        spk2spkix = {int(k): v for v, k in enumerate(spkix2spk)}
    elif spk2spkix is not None and spkix2spk is None:
        spkix2spk = np.zeros(len(spk2spkix))
        for i_, v_ in spk2spkix.items():
            spkix2spk[v_] = i_
    elif spk2spkix is None and spkix2spk is not None:
        spk2spkix = {int(k): v for v, k in enumerate(spkix2spk)}
    else:
        pass

    for spk in spk2spkix:
        y_spk[y_spk == spk] = spk2spkix[spk]
    return y_spk, spkix2spk, spk2spkix


def make_classification_data(model, processor, layer_id, cache_dir,
                             alignment_context_window=[0, 0],
                             alignment_ground_truth_shift=0,
                             data_include_dev_other=False,
                             data_get_only_dev_other=False,
                             data_train_test_spk_overlap=False,
                             norm_spk_index=False,
                             split_ratio=0.2,
                             data_subsets=None,
                             split="dev", preserve_sequence=False):

    spkix2spk, spk2spkix = None, None

    if data_subsets is not None:
        print(f"!! overwriting data_include_dev_other and data_get_only_dev_other using data_subsets={data_subsets} params")
        data_subsets = ",".join(sorted(data_subsets.split(",")))
        data_include_dev_other = None
        data_get_only_dev_other = None
    else:
        data_subsets = ""
        if data_include_dev_other:
            data_subsets = "clean,other"
        elif data_get_only_dev_other:
            data_subsets = "other"
        else:
            data_subsets = "clean"

    print(f"!! Using data_subsets={data_subsets}")

    if "clean" in data_subsets:
        X, y_phone, y_spk = retrive_librispeech_data(model, processor, cache_dir, layer_id, subset="clean",
                                                     alignment_context_window=alignment_context_window,
                                                     alignment_ground_truth_shift=alignment_ground_truth_shift,
                                                     split=split, preserve_sequence=preserve_sequence)
    if "other" in data_subsets:
        X_other, y_phone_other, y_spk_other = retrive_librispeech_data(model, processor, cache_dir, layer_id, subset="other",
                                                                       alignment_context_window=alignment_context_window,
                                                                       alignment_ground_truth_shift=alignment_ground_truth_shift,
                                                                       split=split, preserve_sequence=preserve_sequence)

    if norm_spk_index:
        if data_subsets == "clean":
            y_spk, spkix2spk, spk2spkix = normalize_spk_index(y_spk)
        elif data_subsets == "other":
            y_spk_other, spkix2spk, spk2spkix = normalize_spk_index(y_spk_other)
        elif data_subsets == "clean,other":
            _, spkix2spk, spk2spkix = normalize_spk_index(np.concatenate([y_spk, y_spk_other], axis=0))
            y_spk, _, _ = normalize_spk_index(y_spk, spkix2spk, spk2spkix)
            y_spk_other, _, _ = normalize_spk_index(y_spk_other, spkix2spk, spk2spkix)
        else:
            raise ValueError(f"data_subsets: {data_subsets}")

    if split_ratio is not None:
        if alignment_context_window != [0, 0]:
            alignment_context_window_key = "_contextwin-" + "-".join([str(x) for x in alignment_context_window])
        else:
            alignment_context_window_key = ""

        if "clean" in data_subsets:
            if data_train_test_spk_overlap:
                X_train, y_phone_train, y_spk_train, X_test, y_phone_test, y_spk_test = train_test_split_ensure_spk_overlap(
                    X, y_phone, y_spk, split_test_ratio=split_ratio,
                    spk_choice_file=os.path.join(
                        SPK_TEST_SPLIT_DIR,
                        f"spk_choices_for_test_clean{alignment_context_window_key}.pk",
                    ),
                    preserve_sequence=preserve_sequence)
            else:
                X_train, y_phone_train, y_spk_train, X_test, y_phone_test, y_spk_test = train_test_split(
                    X, y_phone, y_spk, split_ratio=split_ratio,
                    leave_spk_out=not data_train_test_spk_overlap,
                    split_index_file=f"data/train_test_split{alignment_context_window_key}.npy")

        if "other" in data_subsets:
            if data_train_test_spk_overlap:
                X_train_other, y_phone_train_other, y_spk_train_other, X_test_other, y_phone_test_other, y_spk_test_other = train_test_split_ensure_spk_overlap(
                    X_other, y_phone_other, y_spk_other, split_test_ratio=split_ratio,
                    spk_choice_file=os.path.join(
                        SPK_TEST_SPLIT_DIR,
                        f"spk_choices_for_test_other{alignment_context_window_key}.pk",
                    ),
                    preserve_sequence=preserve_sequence)
            else:
                X_train_other, y_phone_train_other, y_spk_train_other, X_test_other, y_phone_test_other, y_spk_test_other = train_test_split(
                    X_other, y_phone_other, y_spk_other, split_ratio=split_ratio,
                    leave_spk_out=not data_train_test_spk_overlap,
                    split_index_file=f"data/train_test_split_other{alignment_context_window_key}.npy")

        if data_subsets == "clean,other":
            if preserve_sequence:
                X_train = X_train + X_train_other
                y_phone_train = y_phone_train + y_phone_train_other
                y_spk_train = y_spk_train + y_spk_train_other
                X_test = X_test + X_test_other
                y_phone_test = y_phone_test + y_phone_test_other
                y_spk_test = y_spk_test + y_spk_test_other
            else:
                X_train = np.concatenate([X_train, X_train_other], axis=0)
                y_phone_train = np.concatenate([y_phone_train, y_phone_train_other], axis=0)
                y_spk_train = np.concatenate([y_spk_train, y_spk_train_other], axis=0)
                X_test = np.concatenate([X_test, X_test_other], axis=0)
                y_phone_test = np.concatenate([y_phone_test, y_phone_test_other], axis=0)
                y_spk_test = np.concatenate([y_spk_test, y_spk_test_other], axis=0)

        if data_subsets in ["clean", "clean,other"]:
            return X_train, y_phone_train, y_spk_train, X_test, y_phone_test, y_spk_test, spkix2spk, spk2spkix
        else:
            return X_train_other, y_phone_train_other, y_spk_train_other, X_test_other, y_phone_test_other, y_spk_test_other, spkix2spk, spk2spkix
    else:
        if data_subsets == "clean,other":
            X = np.concatenate([X, X_other], axis=0)
            y_phone = np.concatenate([y_phone, y_phone_other], axis=0)
            y_spk = np.concatenate([y_spk, y_spk_other], axis=0)

        if data_subsets in ["clean", "clean,other"]:
            return X, y_phone, y_spk
        else:
            return X_other, y_phone_other, y_spk_other
        

def train_test_split_ensure_spk_overlap(X, y_phone, y_spk, split_test_ratio=0.3,
                                        spk_choice_file=os.path.join(
                                            SPK_TEST_SPLIT_DIR,
                                            "spk_choices_for_test_clean.pk",
                                        ),
                                        preserve_sequence=False):
    os.makedirs(os.path.dirname(spk_choice_file), exist_ok=True)

    X_dev, X_test = [], []
    y_phone_dev, y_phone_test = [], []
    y_spk_dev, y_spk_test = [], []

    if preserve_sequence:
        # dict of spk (uniq_spk -> list of utts spoken by spk)
        unique_spk = np.unique([y[0] for y in y_spk])
        test_spk_split = int(len(unique_spk) * split_test_ratio)
        dev_spk = unique_spk[:-test_spk_split]
        test_spk = unique_spk[-test_spk_split:]

        for i in range(len(X)):
            if y_spk[i][0] in dev_spk:
                X_dev.append(X[i])
                y_spk_dev.append(y_spk[i])
                y_phone_dev.append(y_phone[i])
            else:
                X_test.append(X[i])
                y_spk_test.append(y_spk[i])
                y_phone_test.append(y_phone[i])

        assert np.allclose(test_spk, np.unique([y[0] for y in y_spk_test]))
        assert np.allclose(dev_spk, np.unique([y[0] for y in y_spk_dev]))
        assert len(X_dev) + len(X_test) == len(X)

    else:
        if spk_choice_file is not None and os.path.exists(spk_choice_file):
            # all_choices indicates which frames are used for test set
            # boolean dict (nspk -> nframes)
            with open(spk_choice_file, "rb") as out_f:
                all_choices = pickle.load(out_f)
        else:
            all_choices = dict()

        unique_spk = np.unique(y_spk)
        for spk in unique_spk:
            N = len(X[y_spk == spk])
            choice_ = all_choices[spk] if spk in all_choices else None
            if N > 0:
                if choice_ is None:
                    choice_ = np.random.choice(np.arange(N), size=int(split_test_ratio * N), replace=False)
                    all_choices[spk] = choice_
                select_ = np.zeros(N, dtype=np.bool_)
                select_[choice_] = 1
                X_dev.append(X[y_spk == spk][~select_])
                X_test.append(X[y_spk == spk][select_])
                y_spk_dev.append([spk] * (N - len(choice_)))
                y_spk_test.append([spk] * len(choice_))
                y_phone_dev.append(y_phone[y_spk == spk][~select_])
                y_phone_test.append(y_phone[y_spk == spk][select_])

        if spk_choice_file is not None and not os.path.exists(spk_choice_file):
            with open(spk_choice_file, "wb") as out_f:
                pickle.dump(all_choices, out_f)

        X_dev = np.concatenate(X_dev)
        y_phone_dev = np.concatenate(y_phone_dev)
        y_spk_dev = np.concatenate(y_spk_dev)

        X_test = np.concatenate(X_test)
        y_phone_test = np.concatenate(y_phone_test)
        y_spk_test = np.concatenate(y_spk_test)

    return X_dev, y_phone_dev, y_spk_dev, X_test, y_phone_test, y_spk_test
