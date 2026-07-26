# coding=utf-8
"""Transcribe every local clone voice with Whisper large-v3.

The script checkpoints after each clip so it can safely resume. Its JSON output
is later merged into assets/audio/bailian/voices.tsv for the web UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


LANGUAGE_HINTS = {
    "loongindah_v3.wav": "indonesian",
    "loongyuuna_v3.wav": "japanese",
    "loongriko_v3.wav": "japanese",
    "loongkyong_v3.wav": "korean",
    "loongcally_v3.wav": "english",
}
LANGUAGE_LABELS = {
    "chinese": "Chinese",
    "english": "English",
    "japanese": "Japanese",
    "korean": "Korean",
    "indonesian": "Indonesian",
}
DISPLAY_LANGUAGE_HINTS = {
    "longjiayi_v2.mp3": "Cantonese",
    "longanmin.mp3": "Chinese (Minnan)",
    "qwen3_mia.wav": "English",
}
TRANSCRIPT_CORRECTIONS = {
    "longdaiyu.mp3": "往日里，哥哥总是忙里偷闲地来敷衍我。谁承想，今儿起了个大早，网也不上了，球也不打了，巴巴地凑了过来。",
    "longwanjun.mp3": "苏婉坐在窗边，手指轻轻摩挲着那本泛黄的日记。纸页上娟秀的字迹还带着当年的温度。她想起昨天在老宅阁楼里发现它时的场景：日记本被藏在一个旧木盒里，旁边还放着一枚生锈的银簪，簪头刻着小小的“尘”字。",
    "longyue_v2.mp3": "当笛卡尔的《我思故我在》与庄子的《庄周梦蝶》在空气中交织，思想的火花点燃了声音的舞台。这不是枯燥的说教，而是哲人用声音编织的思维迷宫。",
    "longqiang_v3.mp3": "宝儿，今天下班快回来吧，我做了你最爱吃的红烧肉，再不回来，菜都要凉了。",
    "longanmin.mp3": "哇，你看这个小蛋糕好可爱哦，上面还有一个小兔子造型耶。看起来还软软的，我们买一个回去好不好？看起来真好吃的样子。",
    "longjiayi_v2.mp3": "上班再忙，都不要忘记照顾自己，记得多饮啲水，食啲好嘢。",
    "longhua_v3.mp3": "一起加油吧，明天要去哪？咱们去喝杯奶茶冷静下，明天继续赶工也来得及。",
    "longmiao_v3.mp3": "月光洒在泛黄的纸页上，诗句化作溪流在耳畔潺潺。听徐志摩的柔情在康桥泛起涟漪，让《海子诗选》的麦浪与《陶渊明集》的南山，在声音的褶皱里悄然生长。",
    "loongkyong_v3.wav": "저는 운전을 하다 길을 잃어서 같은 길을 빙빙 돌아서 목적지에 도착했어요.",
    "longanqin.mp3": "诶，你们听说了吗？楼下新开一家奶茶店，装修得特别清新，还有好多新奇的口味，像什么杨枝甘露爆珠茶、草莓奶冻茶，听着就想喝。咱们下班一起去试试呗，要是好喝的话，以后下午茶就有着落了。",
    "qwen3_elias.wav": "对，那在古典文学当中呢，我们说这个香兰杜若啊，经常是会象征这个才子他的一个孤高和无奈。比如说在这个屈原的《离骚》当中就提起过，花开花落，它暗喻了人生的起伏。",
    "qwen3_chelsie.wav": "还让不让人好好减肥了？不行，你要请我一顿好的赔偿我。",
    "qwen3_stella.wav": "哇，真的呢，面包香喷喷的，好想吃一口呀。不过，那个，作为越野兔，我得保持警惕。小镇这么和平，可不能有坏人捣乱，懂吧？",
    "qwen3_vivian.wav": "铜锣湾只有一个浩南，导航界只有我这个大姐。系好安全带啊，这条路今天我罩你啊！这条路走了三年了，不想证明什么，我是想让你知道，你走错路了，大佬。重新规划吧，这次我帮你摆平。",
    "qwen3_mia.wav": "Hey friends, welcome back. I hope you're doing well today, and that wherever you're watching from, you're feeling a little warm, a little calm, maybe even holding your favorite drink. You know, something soft to settle into the day with.",
    "qwen3_maia.wav": "我？你怎么知道我挺会背诗的？你是不是对我有一些了解？我高中语文也不是特别好吧，大概也就是全年级前十吧。",
}


def normalize_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    return text


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        mono = resample_poly(
            mono,
            target_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(mono)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"model": "", "updated_at": 0.0, "results": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_voices_tsv(path: Path, rows: list[dict[str, str]], results: dict) -> None:
    fieldnames = [
        "name",
        "description",
        "filename",
        "source_url",
        "language",
        "transcript",
        "transcript_source",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            filename = str(row.get("filename") or "")
            result = results[filename]
            output_row = dict(row)
            output_row["language"] = DISPLAY_LANGUAGE_HINTS.get(
                filename,
                LANGUAGE_LABELS.get(
                    str(result.get("language") or "chinese"),
                    str(result.get("language") or "Chinese").title(),
                ),
            )
            output_row["transcript"] = normalize_transcript(
                TRANSCRIPT_CORRECTIONS.get(filename, result.get("transcript", ""))
            )
            output_row["transcript_source"] = (
                "Whisper large-v3 + manual review"
                if filename in TRANSCRIPT_CORRECTIONS
                else "Whisper large-v3"
            )
            writer.writerow(output_row)
    temporary.replace(path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/Whisper-large-v3"),
    )
    parser.add_argument(
        "--voices-tsv",
        type=Path,
        default=Path("assets/audio/bailian/voices.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/audio/bailian/transcripts.auto.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    voices_tsv = args.voices_tsv.resolve()
    output = args.output.resolve()
    audio_dir = voices_tsv.parent

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(model_dir)
    checkpoint = load_checkpoint(output)
    checkpoint["model"] = str(model_dir)
    results = checkpoint.setdefault("results", {})
    rows = load_rows(voices_tsv)

    for index, row in enumerate(rows, start=1):
        filename = str(row.get("filename") or "").strip()
        if not filename:
            continue
        if not args.force and filename in results and results[filename].get("transcript"):
            print(f"[{index:02d}/{len(rows)}] cached {filename}", flush=True)
            continue
        audio_path = audio_dir / filename
        language = LANGUAGE_HINTS.get(filename, "chinese")
        started = time.perf_counter()
        audio = load_audio(audio_path, processor.feature_extractor.sampling_rate)
        features = processor(
            audio,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        input_features = features.input_features.to(args.device, dtype=dtype)
        with torch.inference_mode():
            generated_ids = model.generate(
                input_features=input_features,
                language=language,
                task="transcribe",
            )
        transcript = normalize_transcript(
            processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        )
        results[filename] = {
            "name": row.get("name", ""),
            "language": language,
            "transcript": transcript,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        checkpoint["updated_at"] = time.time()
        save_checkpoint(output, checkpoint)
        print(
            f"[{index:02d}/{len(rows)}] {filename} [{language}] {transcript}",
            flush=True,
        )
    missing = [
        str(row.get("filename") or "")
        for row in rows
        if not normalize_transcript(results.get(str(row.get("filename") or ""), {}).get("transcript", ""))
    ]
    if missing:
        raise RuntimeError(f"Transcription is incomplete: {', '.join(missing)}")
    update_voices_tsv(voices_tsv, rows, results)
    print(f"Updated {voices_tsv} with {len(rows)} ICL transcripts.", flush=True)


if __name__ == "__main__":
    main()
