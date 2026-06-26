import torch
from datasets import load_dataset
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import numpy as np
import gzip, pickle, os 
from einops import rearrange, reduce
from transformers import (
    HubertModel,
    HubertForCTC,
)
from concept_space import utils
from joblib import Parallel, delayed
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(dataset, model, processor, cache_dir=None, return_hidden=True, layer_id=None, model_mode=None):
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
 
    if type(model) == HubertForCTC:
        return inference_handler(dataset, model.hubert, processor, cache_dir, return_hidden, layer_id)
    elif type(model) == HubertModel:
        return inference_handler(dataset, model, processor, cache_dir, return_hidden, layer_id)
    else:
        raise ValueError("Unknown model")


def inference_melspectrogram(dataset, sampling_rate=16000, win_length_ms=25, hop_length_ms=20, is_add_post_layer_norm=True):

    def moving_average(x, window_size=[3], stride=[1], padding=None):
        x_in = rearrange(x, 'dim seq_len -> seq_len dim')
        # if padding is not None:
        #     x_in = np.concatenate([x_in, np.full( (stride, x.shape[-1]), padding)]    
        for win_size_, stride_ in zip(window_size, stride):
            out = []
            for i in range(0, len(x_in)-(win_size_-1), stride_):
                out.append(
                    reduce(x_in[i:i+win_size_], 'window_size dim -> dim', 'mean')
                )
            x_in = out
        out = np.asarray(out)
        assert np.isnan(out).any() == False
        return rearrange( np.asarray(out), 'seq_len dim -> dim seq_len' )

    mel_spectrogram_params = {
        "sample_rate": sampling_rate,
        "n_fft" :512,
        "win_length" :int(sampling_rate * win_length_ms/1000), # 25ms windown size
        "hop_length" :int(sampling_rate * hop_length_ms/1000), # 20ms windown size
        "n_mels" :80,
    }
    moving_avg_params = None
    # moving_avg_params = {
    #     "window_size": [3, 3],
    #     "stride": [1, 2],
    #     "padding": None,
    # }
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        **mel_spectrogram_params
    )
    subsampling_factor = mel_spectrogram_params["hop_length"]
    if moving_avg_params is not None:
        moving_avg_fn = lambda x: moving_average(x, **moving_avg_params)
        for s in moving_avg_params["stride"]:
            subsampling_factor *= s
    else:
        moving_avg_fn = lambda x: x

    if is_add_post_layer_norm:
        layer_norm = torch.nn.LayerNorm(mel_spectrogram_params["n_mels"])

    features = []
    with torch.no_grad():
        for i in range(len(dataset)):
            mel = mel_spectrogram(torch.tensor(dataset[i]["audio"]["array"], dtype=torch.float32))
            if is_add_post_layer_norm:
                mel = layer_norm(mel.T).T
            features.append(moving_avg_fn(
                mel.numpy()
            ))
        id2logit = { 
            dataset["id"][i]: features[i].T
            for i in range(len(dataset["id"])) 
        }
    
    return id2logit
    


def inference_handler(dataset, model, processor, cache_dir=None, return_hidden=True, layer_id=None):
    # hidden_states_out = []
    sampling_rate = utils.get_sampling_rate_from_processor(processor)

    for sample in tqdm(dataset):
        inputs = processor(sample["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

        # support both transformers Processor and FeatureExtractor classes
        input_values = getattr(inputs, "input_values", None)
        if input_values is None:
            input_values = getattr(inputs, "input_features", None)
        if input_values is None:
            raise ValueError("Processor output does not contain 'input_values' or 'input_features'.")

        with torch.no_grad():
            _outputs = model(
                input_values.to(model.device),
                output_hidden_states=True
            )
        outputs = tuple([rearrange(h.to('cpu').float().numpy(), '() t h -> t h') for h in _outputs.hidden_states])

        if cache_dir is not None:
            for layer_i, output in enumerate(outputs):                
                # if layer_id is not None and layer_i != layer_id:
                #     continue
                if "speaker_id" not in sample:
                    speaker_id = "unknown_speaker"
                else:
                    speaker_id = str(sample["speaker_id"])

                cache_path = os.path.join(cache_dir, speaker_id, sample["id"], f"{layer_i}.npy")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, output)
        del outputs
        del _outputs
        # if return_hidden:
        #     hidden_states_out.append(
        #         outputs
        #     )
        # else:
        #     outputs = _outputs.last_hidden_state.to('cpu').numpy().reshape(-1, hdim)
        #     hidden_states_out.append(outputs)
    # return hidden_states_out


def retrive_logit_mp(path):
    if not os.path.exists(path):
        path = path.replace(".npy", ".pk")
    
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".pk"):
        with open(path, "rb") as in_f:
            _logit = pickle.load(in_f)
            return _logit
    else:
        raise ValueError("Unsupported file format")


def retrive_logit_at_layer(id, layer_idx, cache_dir, unknown_spk=False):
    # hidden_states: [ [ tensor(T x hdim) ] x layers ] x samples
    if type(id) == str:
        id = [id]    
    cache_files = []
    for _id in id:
        if unknown_spk:
            spk_id = "unknown_speaker"
        else:
            spk_id = _id.split("-")[0]
        cache_files.append(os.path.join(cache_dir, spk_id, _id, f"{layer_idx}.npy"))

    results = Parallel(n_jobs=4, verbose=0, timeout=99999)(
        delayed(retrive_logit_mp)(
            f, 
        ) for f in cache_files
    )
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-compute HuBERT hidden states for LibriSpeech")
    parser.add_argument("--cache-dir", default="cache", help="Root directory for cached features")
    args = parser.parse_args()

    model_uri = "facebook/hubert-base-ls960"
    model_name = model_uri.split("/")[-1]
    cache_dir = os.path.join(args.cache_dir, model_name, "Librispeech")

    processor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True)

    print(f"Loading {model_uri}...")
    model = AutoModel.from_pretrained(model_uri)
    model.eval()

    print(f"Processing librispeech_asr clean/validation -> {cache_dir}")
    dataset = load_dataset("openslr/librispeech_asr", name="clean", split="validation", trust_remote_code=True)
    inference_handler(dataset, model, processor, cache_dir=cache_dir)
    print("Done")


if __name__ == "__main__":
    main()

