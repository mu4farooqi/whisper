import argparse
import time
import os
import cupy as cp
import warnings
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import tqdm
import string
from dtw import dtw
from cupyx.scipy.signal import medfilt

from .audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram, load_audio
from .decoding import DecodingOptions, DecodingResult
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt

if TYPE_CHECKING:
    from .model import Whisper


def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    gpu_memory_fraction: float = 0.85,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    
    gpu_memory_fraction: float
        defined what fraction of gpu memory transcribe/translate operation should use. If memory usage
        exceeds this fraction, pytorch will free some memory from its cache to fix fragmentation.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False
        
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, 0)

    # Removes silences from the audio file. This results in better transcription quality
    # without hallucinations.
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True, trust_repo=True)
    (get_speech_timestamps, _, read_audio, _, collect_chunks) = utils
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = read_audio(audio, sampling_rate=SAMPLE_RATE)
        else:
            audio = torch.from_numpy(audio)

    speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=SAMPLE_RATE, window_size_samples=512, speech_pad_ms=45, threshold=0.5)
    audio = collect_chunks(speech_timestamps, audio)
    
    mel = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)
    
    def split_tokens_on_unicode(tokens: torch.Tensor):
        words = []
        word_tokens = []
        current_tokens = []

        for token in tokens.tolist():
            current_tokens.append(token)
            decoded = tokenizer.decode_with_timestamps(current_tokens)
            if "\ufffd" not in decoded:
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []

        return words, word_tokens

    def split_tokens_on_spaces(tokens: torch.Tensor):
        subwords, subword_tokens_list = split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= tokenizer.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens
    
    if LANGUAGES[language].title() in {"Chinese", "Japanese", "Thai", "Lao", "Myanmar"}:
        # These languages don't typically use spaces, so it is difficult to split words
        # without morpheme analysis. Here, we instead split words at any
        # position where the tokens are decoded as valid unicode points
        split_tokens = split_tokens_on_unicode
    else:
        split_tokens = split_tokens_on_spaces

        
    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
                needs_fallback = True  # too repetitive
            if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result
    
    # Install hooks on the cross attention layers to retrieve the attention weights 
    QKs = [None] * model.dims.n_text_layer

    for i, block in enumerate(model.decoder.blocks):
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    samples_per_token = int(input_stride * HOP_LENGTH)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0
    audio_duration_in_seconds = round(len(audio) / SAMPLE_RATE, 3)
    medfilt_width = 7
    qk_scale = 1.0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
        *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": text_tokens.tolist(),
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            uncasted_segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device)
            segment = uncasted_segment.to(dtype)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(segment)
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                text_tokens = []
                for idx, current_slice in enumerate(consecutive):
                    sliced_tokens = tokens[last_slice:current_slice]                    
                    if idx == 0:
                        start_timestamp_position = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                        )
                        start_timestamp = timestamp_offset + start_timestamp_position * time_precision
                    text_tokens.append(sliced_tokens[1:-1]),
                    last_slice = current_slice
                
                last_timestamp_position = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                end_timestamp = min(timestamp_offset + last_timestamp_position * time_precision, audio_duration_in_seconds)

                add_segment(
                    start=start_timestamp,
                    end=end_timestamp,
                    text_tokens=torch.cat(text_tokens),
                    result=result,
                )
                
                duration = end_timestamp - start_timestamp
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                duration = min(duration, audio_duration_in_seconds - timestamp_offset)
                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())
                
            ideal_number_of_tokens = int(duration * SAMPLE_RATE // samples_per_token)
            tokens_ts = torch.tensor(
                [
                    *tokenizer.sot_sequence,
                    tokenizer.timestamp_begin,
                ] + tokenizer.encode(all_segments[-1]['text']) + [
                    tokenizer.timestamp_begin + ideal_number_of_tokens,
                    tokenizer.eot,
                ]
            ).cuda().long()
            words, word_tokens = split_tokens(tokens_ts)
            
            with torch.no_grad():
                logits = model(uncasted_segment.unsqueeze(0), tokens_ts.unsqueeze(0))
                
            del tokens_ts, logits

            qk_weights = torch.concatenate(QKs)  # layers * heads * tokens * frames
            cp_weights = cp.asarray(qk_weights[:, :, :, : ideal_number_of_tokens])
            medfilter_weights = medfilt(cp_weights, (1, 1, 1, medfilt_width))
            weights = torch.as_tensor(medfilter_weights, device='cpu') * qk_scale
            weights = weights.clone().detach().softmax(dim=-1)

            w = weights / weights.norm(dim=-2, keepdim=True)
            matrix = w[-6:].mean(axis=(0, 1))

            alignment = dtw(-matrix.double().numpy())

            jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
            jump_times = alignment.index2s[jumps] * time_precision
            
            word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
            begin_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]

            all_segments[-1]['word_timestamps'] = [
                dict(word=word, begin=timestamp_offset + begin, end=timestamp_offset + end)
                for word, begin, end in zip(words[:-1], begin_times, end_times)
                if not word.startswith("<|") and word.strip() not in "、。"
            ]
            
            # Free-up pytorch cache memory
            del qk_weights, cp_weights, medfilter_weights, weights

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)


def cli():
    from . import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    from . import load_model
    model = load_model(model_name, device=device, download_root=model_dir)

    for audio_path in args.pop("audio"):
        result = transcribe(model, audio_path, temperature=temperature, **args)

        audio_basename = os.path.basename(audio_path)

        # save TXT
        with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
            write_txt(result["segments"], file=txt)

        # save VTT
        with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
            write_vtt(result["segments"], file=vtt)

        # save SRT
        with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)


if __name__ == '__main__':
    cli()
