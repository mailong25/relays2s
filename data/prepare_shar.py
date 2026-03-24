import argparse
import json
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from lhotse import CutSet, Recording, SupervisionSegment, AudioSource
from lhotse.shar import ArrayTarWriter
from modules.feature_extractor import AudioFeatureExtractor
from functools import partial
import os

def process_cut(cut, noise_paths=None):
    audio_path = cut.recording.sources[0].source
    feature_extractor = AudioFeatureExtractor()
    feats = feature_extractor.extract_offline(audio_path, noise_paths).numpy()
    
    cut = cut.attach_tensor(
        "fbank", feats,
        frame_shift=float(feature_extractor.frame_shift) / 1000,
        temporal_dim=0,
    )
    return cut.id, feats, cut.fbank

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to the input dataset.jsonl file."
    )
    parser.add_argument(
        "--audios-dir",
        type=str,
        required=True,
        help="Path to the directory containing audio files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the Shar dataset."
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Number of cuts per shard."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers for FBANK extraction."
    )
    parser.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        help="If provided, augment the conversation with noises from noise_dir"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = [json.loads(line) for line in open(args.input_jsonl)]
    
    all_cuts = []

    for sample in tqdm(data, desc="Preparing cuts"):
        user_audio = sample['user_audio']
        user_label = sample['user_label']
        assistant_label = sample['assistant_label']
        
        user_audio = os.path.join(args.audios_dir, user_audio)
        # Only user audio is needed as input
        recording_user = Recording(
            id=sample['conv_id'] + '_user',
            sources=[AudioSource(source=user_audio, channels=[0], type='file')],
            sampling_rate=sample['sampling_rate'],
            num_samples=sample['num_samples'],
            duration=sample['conv_len'],
        )
        
        # Assistant_text is needed as target output
        supervisions = []
        for idx, turn in enumerate(assistant_label):
            supervisions.append(
                SupervisionSegment(
                    id=f"{sample['conv_id']}_assistant_turn_{idx}",
                    recording_id=recording_user.id,
                    start=turn['start'],
                    duration=turn['end'] - turn['start'],
                    text=turn['text'],
                    speaker="assistant",
                    custom={"type": turn.get('type','')},
                )
            )
        
        # User_text is needed to pretrain Speech Adapter
        for idx, turn in enumerate(user_label):
            supervisions.append(
                SupervisionSegment(
                    id=f"{sample['conv_id']}_user_turn_{idx}",
                    recording_id=recording_user.id,
                    start=turn['start'],
                    duration=turn['end'] - turn['start'],
                    text=turn['text'],
                    speaker="user",
                    custom={"type": turn.get('type','')},
                )
            )
        
        cut = recording_user.to_cut()
        cut.supervisions = supervisions
        all_cuts.append(cut)

    cuts = CutSet(all_cuts)

    # --- Step 2: Save cuts/recordings into Shar shards ---
    shards = cuts.to_shar(
        output_dir,
        fields = {},
        shard_size=args.shard_size
    )

    if args.noise_dir:
        noise_paths = [os.path.join(args.noise_dir, f) for f in os.listdir(args.noise_dir)]
        print("Found", len(noise_paths), "noise clips")
    else:
        noise_paths = None
    
    worker_fn = partial(process_cut, noise_paths=noise_paths)
    
    # --- Step 3: Compute FBANKs in parallel and save as extra shar field ---
    with ArrayTarWriter(
        f"{output_dir}/fbank.%06d.tar",
        shard_size=args.shard_size,
        compression="lilcom"
    ) as writer:
        with mp.Pool(processes=args.num_workers) as pool:
            for cut_id, feats, feat_manifest in tqdm(
                pool.imap(worker_fn, cuts, chunksize=32),
                total=len(cuts),
                desc="Extracting fbank (ordered)"
            ):
                writer.write(cut_id, feats, feat_manifest)

    print("Shar dataset prepared at:", output_dir)