# KidSpeak: A General Multi-purpose LLM for Kids' Speech Recognition and Screening

[![arXiv](https://img.shields.io/badge/arXiv-2512.05994-b31b1b.svg)](https://arxiv.org/abs/2512.05994)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-KidSpeak--Instruct-yellow)](https://huggingface.co/datasets/jsun39/KidSpeak-Instruct)
[![LLM](https://img.shields.io/badge/🤗%20Model-kidspeak__vicuna-yellow)](https://huggingface.co/jsun39/kidspeak_vicuna)

<p align="center"><img src="assets/kidspeak.png" width="720"></p>

KidSpeak is a speech LLM for children's speech. Given an audio clip and a question, it answers in
natural language. One model handles several tasks:

| Task | Example question | Metric |
|---|---|---|
| Speech description / gender | *Describe what you hear in the clip.* | accuracy |
| Speech-disorder screening (binary) | *Is the child's speech typical or impaired?* | accuracy |
| Disorder type (6 SSD subtypes) | *What category of disorder does it fall into?* | accuracy |
| Age prediction | *Can you tell the age of the speaker?* | exact / age-group accuracy |
| Transcription | *Transcribe the speech.* | WER / CER |
| Dialect (native / non-native UK) | *Identify the speaker's dialect by their accent nuances.* | accuracy |

## Method

```
audio ─► Whisper encoder (frozen) ─► [B,1500,D] ─► stack 6 frames ─► [B,250,6D] ─► Linear ─► [B,250,4096]
                                                                                            │
 "<s>### Human: <Img>" + [250 audio tokens] + "</Img> {question}\n### Assistant:" ─► Vicuna-7B + LoRA ─► answer
```

* **Audio encoder:** OpenAI Whisper encoder (`tiny` … `large-v3`), kept frozen.
* **Projection:** six consecutive Whisper frames are concatenated into one token, which gives 250 tokens
  per 30 s clip. A single linear layer then maps each token to the LLM hidden size.
* **LLM:** Vicuna-7B v0 ([`jsun39/kidspeak_vicuna`](https://huggingface.co/jsun39/kidspeak_vicuna)) with
  LoRA (r=16, α=32) on `q/k/v/o_proj`.
* **Training:** multi-turn instruction tuning with cross-entropy on the assistant turns only. Only the
  LoRA adapters and the projection are trained (~70 MB per checkpoint with Whisper-small).

## Repository structure

```
KidSpeak/
├── train.py              # DeepSpeed instruction tuning
├── inference.py          # multi-turn QA on a test split  -> predictions.jsonl
├── compute_metrics.py    # predictions.jsonl -> per-task metrics
├── kidspeak/
│   ├── model.py          # KidSpeak model (Whisper + projection + LoRA Vicuna)
│   ├── audio.py          # audio loading / log-Mel features
│   ├── dataset.py        # instruction dataset
│   └── modeling_llama.py # LLaMA implementation (from HF transformers)
├── configs/
│   ├── kidspeak.yaml     # model / LoRA / epochs
│   └── ds_config.json    # DeepSpeed: batch size, lr, optimizer, bf16
├── data_prep/
│   ├── build_dataset.py  # raw corpora -> instruction JSON (optional)
│   └── prompts/          # question / answer templates
└── scripts/
    ├── train.sh
    └── eval.sh
```

## 1. Installation

Tested with Python 3.10, CUDA 11.8, PyTorch 2.4 and 4× A6000 / A100 GPUs.

```bash
conda create -y -p envs/kidspeak python=3.10     # the environment lives inside the repo
conda activate ./envs/kidspeak
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 2. Data

### 2.1 Audio

Download the raw audio and put it under `dataset/KIDS/`. The JSON files refer to audio by paths relative
to `dataset/`.

```
dataset/KIDS/
├── ultrasuite_disorder/        # UltraSuite UPX (speech sound disorders)
│   └── core-upx/{core,doc}/
├── talkbank_dataset/v1.3/official_v1.3/
│   ├── talkbank_childes.csv
│   └── usable/FASA_ENNI/out/<id>/*.mp3
└── english_children/           # Kennedy et al. (HRI 2017), native / non-native UK children
    ├── english_free_speech/files_cut_by_sentences/
    └── english_words_sentences/
```

* **UltraSuite UPX:** get it from the [UltraSuite website](https://ultrasuite.github.io/download/). Only
  `core-upx` is needed (~191 GB):
  `mkdir -p dataset/KIDS/ultrasuite_disorder && rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-upx dataset/KIDS/ultrasuite_disorder/`
* **English children:** this is the child speech corpus from Kennedy et al., *Child Speech Recognition in
  Human-Robot Interaction: Evaluations and Recommendations* (HRI 2017), licensed CC-BY 4.0. Download it
  from [Zenodo](https://zenodo.org/records/200495) and unzip it into `dataset/KIDS/`:
  `wget https://zenodo.org/records/200495/files/english_children.zip && unzip english_children.zip -d dataset/KIDS/`
* **TalkBank ENNI:** the audio and transcripts come from the
  [ENNI corpus](https://talkbank.org/childes/access/Clinical-Eng/ENNI.html) in CHILDES. They are segmented
  into utterances with our aligner **FASA** (see the paper). You need to follow the TalkBank usage rules.

### 2.2 Instruction JSON

Download the instruction-tuning files used in the paper from Hugging Face:

```bash
huggingface-cli download jsun39/KidSpeak-Instruct --repo-type dataset --local-dir dataset/json
```

| split | UltraSuite | ENNI | English children | merged |
|---|---:|---:|---:|---:|
| train | 1,952 | 10,257 | 190 | 12,399 |
| val | 558 | 2,931 | 54 | 3,543 |
| test | 279 | 1,466 | 28 | 1,773 |

Each sample is one audio clip with a multi-turn conversation:

```json
{
  "audio_name": "KIDS/ultrasuite_disorder/core-upx/core/19M/BL1/006A.wav",
  "conversation": [
    {"from": "human", "value": "Does the speaker in this audio exhibit typical speech or a speech impairment?"},
    {"from": "gpt",   "value": "The speaker has a speech disorder based on the speech you hear."},
    {"from": "human", "value": "Based on the speech you hear, what category of disorder does it fall into?"},
    {"from": "gpt",   "value": "..."}
  ]
}
```

*Optional:* you can rebuild the JSON files from the raw data with
`python data_prep/build_dataset.py --data_root dataset --output_dir dataset/json`. The prompts are sampled at
random, so the rebuilt files will not be identical to the released ones.

## 3. Training

```bash
bash scripts/train.sh small 4 10   # <whisper_model> <num_gpus> <epochs>
```

This runs `train.py` on `dataset/json/merged_train.json` and writes a checkpoint to
`checkpoint/kidspeak_small/pytorch_model_<epoch>.pt` after every epoch. The run config is saved as
`config.yaml` next to the checkpoints.

* The Whisper size is `tiny | base | small | medium | large-v3`.
* Model and LoRA hyper-parameters are in `configs/kidspeak.yaml` (10 epochs by default).
* Batch size, learning rate and precision are in `configs/ds_config.json`. The global batch is 128, with a
  micro-batch of 4 × grad-accum 8 × 4 GPUs. If you use a different number of GPUs, change
  `gradient_accumulation_steps` so that `train_batch_size` stays equal to micro-batch × grad-accum × #GPUs.

## 4. Evaluation

```bash
bash scripts/eval.sh small 9 test  # <whisper_model> <epoch> <split: test | val>
```

For each dataset of the split (UltraSuite, ENNI, English children), the script:

1. runs `inference.py`, which asks every question of each test conversation in turn and writes
   `outputs/kidspeak_small/epoch_9/test/<task>.jsonl`, and
2. runs `compute_metrics.py`, which scores the predictions and writes `<task>.metrics.json`:

```json
{"gender_acc": ..., "binary_disorder_acc": ..., "multi_disorder_acc": ..., "age_acc": ..., "age_group_acc": ..., "num_samples": {...}}
```

`wer` and `cer` are error rates (lower is better). The other metrics are accuracies in %. The age groups
are 0–3, 4–5, 6–8, 9–12 and 13–17 years.

To score a single test file:

```bash
python inference.py --ckpt checkpoint/kidspeak_small/pytorch_model_9.pt \
    --test_file dataset/json/ultrasuite_disorder_test.json --output outputs/ultrasuite.jsonl
python compute_metrics.py --pred outputs/ultrasuite.jsonl
```

## Citation

```bibtex
@article{sharma2025kidspeak,
  title   = {KidSpeak: A General Multi-purpose LLM for Kids' Speech Recognition and Screening},
  author  = {Sharma, Rohan and Liu, Dancheng and Sun, Jingchen and Zhou, Shijie and Qin, Jiayu and Xiong, Jinjun and Chen, Changyou},
  journal = {arXiv preprint arXiv:2512.05994},
  year    = {2025}
}
```

## Acknowledgements

This codebase builds on [PandaGPT](https://github.com/yxuansu/PandaGPT),
[OpenAI Whisper](https://github.com/openai/whisper), [Vicuna](https://github.com/lm-sys/FastChat) and
[PEFT](https://github.com/huggingface/peft). The data comes from [UltraSuite](https://ultrasuite.github.io/),
[TalkBank / CHILDES](https://talkbank.org/) and the [child speech corpus of Kennedy et al. (2017)](https://zenodo.org/records/200495). Please follow the license terms of each dataset.
