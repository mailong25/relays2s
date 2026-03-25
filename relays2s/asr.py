from faster_whisper import WhisperModel
import torch
import time
from pydub import AudioSegment
import os
from tqdm import tqdm
import json
from jiwer import wer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_jsonl", type=str, default="data/test.jsonl")
parser.add_argument("--audio_dir", type=str, default="data/audio")
parser.add_argument("--model_name", type=str, default="medium.en")
parser.add_argument("--save_path", type=str, default="data/asr_map.json")
args = parser.parse_args()

model = WhisperModel(args.model_name, device="cuda", compute_type="float16")

def get_asr(wav_path):
    segments, info = model.transcribe(wav_path, beam_size=5)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip()

# Load JSONL
with open(args.input_jsonl, "r") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

map_asr = {}
refs = []
hyps = []
lats = []

for sample in tqdm(data):
    conv_id = sample["conv_id"]
    audio_path = os.path.join(args.audio_dir, sample["user_audio"])
    audio = AudioSegment.from_wav(audio_path)

    # Combine user and assistant labels, sorted by start time
    all_labels = []
    for label in sample.get("user_label", []):
        all_labels.append({**label, "speaker": "user"})
    for label in sample.get("assistant_label", []):
        all_labels.append({**label, "speaker": "assistant"})
    all_labels.sort(key=lambda x: x["start"])

    acc_contexts = []
    for label in all_labels:
        if label["speaker"] == "assistant":
            if label.get("type", "") != "backchannel":
                acc_contexts.append({"text": label["text"], "latency": None})
        elif label["speaker"] == "user":
            part = audio[max(int(label["start"] * 1000) - 200, 0): min(int(label["end"] * 1000) + 200, len(audio))]
            part.export("temp.wav", format="wav")
            torch.cuda.synchronize()
            start = time.time()
            asr = get_asr("temp.wav")
            acc_contexts.append({"text": asr, "latency": time.time() - start})
            refs.append(label["text"])
            hyps.append(asr)
            lats.append(time.time() - start)
            torch.cuda.synchronize()
    map_asr[conv_id] = acc_contexts

print("WER:", wer(refs, hyps))
print("Average latency per user turn: %.3fs" % (sum(lats) / len(lats)))
json.dump(map_asr, open(args.save_path, "w"), indent=2)