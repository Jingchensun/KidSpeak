"""Build the KidSpeak instruction-tuning JSON files from the raw corpora.

Every corpus is split 7:3 into train / test at the speaker level (stratified by label), so every child
appears in exactly one split.

Expected layout under --data_root (audio_name paths in the JSON are relative to it):

    dataset/KIDS/
    ├── ultrasuite_disorder/                    # official UltraSuite downloads (wav + txt + doc only)
    │   ├── core-upx/{doc, core/<spk>/BL*/*.wav}   # 20 children with SSD (+ subtype)
    │   ├── core-uxssd/{doc, core/<spk>/BL*/*.wav} #  8 children with SSD
    │   ├── core-uxtd/{doc, core/<spk>/*.wav}      # 58 typically developing children
    │   ├── labels/{upx,uxssd,uxtd}/               # UltraSuite labels release (speaker / word labels, transcriptions)
    │   └── child_only/                            # written by this script: clips cropped to the child's speech
    ├── talkbank_dataset/v1.3/official_v1.3/
    │   ├── talkbank_childes.csv                # FASA metadata
    │   └── usable/FASA_ENNI/out/<id>/*.mp3 (+ .txt)          # one folder per child
    └── english_children/                       # english_children.zip from https://zenodo.org/records/200495
        ├── english_free_speech/files_cut_by_sentences/<spk>/*.wav
        └── english_words_sentences/<spk>/**/studio_mic/sentences/*.wav

Outputs (in --output_dir): {ultrasuite,enni,english_children}_{train,test}.json,
merged_{train,test}.json and split_stats.json.
"""
import argparse
import ast
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')
SPLITS = ('train', 'test')


def load_prompts(name):
    with open(os.path.join(PROMPT_DIR, name), encoding='utf-8') as f:
        return [line.strip() for line in f]


P = {k: load_prompts(v) for k, v in {
    'what_you_hear': 'whatyouhear.txt',
    'a_what': 'a_whatyouhear.txt',
    'transcribe': 'transctibe_english.txt',
    'dialect': 'dialect.txt',
    'binary_q': 'disorder_binary_q.txt',
    'binary_yes': 'binary_disorder_binary_isdis.txt',
    'binary_no': 'binary_disorder_binary_nodis.txt',
    'multi_q': 'disorder_multi_q.txt',
    'multi_a': 'a_disorder.txt',
    'age_q': 'age_q.txt',
    'age_a': 'age_a.txt',
    'gender_q': 'gender_q.txt',
    'gender_boy': 'gender_boy.txt',
    'gender_girl': 'gender_girl.txt',
}.items()}


def qa(question_key, answer):
    return [{'from': 'human', 'value': random.choice(P[question_key])}, {'from': 'gpt', 'value': answer}]


def speaker_split(speaker_label, seed, test_ratio=0.3):
    """Stratified speaker-level train / test split. `speaker_label` maps speaker -> stratum (e.g. TD / SLI).

    Within each stratum, round(n * test_ratio) speakers go to test, and at least one if the stratum has
    two or more speakers.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for spk, label in sorted(speaker_label.items()):
        by_label[label].append(spk)
    assignment = {}
    for label in sorted(by_label):
        spks = by_label[label]
        rng.shuffle(spks)
        n = len(spks)
        n_test = max(1, round(n * test_ratio)) if n >= 2 else 0
        for i, spk in enumerate(spks):
            assignment[spk] = 'test' if i < n_test else 'train'
    return assignment


def group(samples, assignment):
    splits = {s: [] for s in SPLITS}
    for sample in samples:
        splits[assignment[sample.pop('speaker')]].append(sample)
    return splits


# ------------------------- UltraSuite (UPX + UXSSD: disorder, UXTD: typical) -------------------------
KEEP_PROMPT_TYPES = {'words', 'sentence', 'non-words'}  # shared by all three subsets; drops articulatory / non-speech
SEGMENT_MARGIN = 0.05  # seconds of context kept around each child segment


def read_lab(path):
    """HTK label file (start / end in units of 100 ns) -> [(start_sec, end_sec, label)]."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(int(a) / 1e7, int(b) / 1e7, lab) for a, b, lab in (line.split() for line in f if line.strip())]


def clean_uxtd_transcription(text):
    """Keep the child's words only. Returns None if the child produced unintelligible / partial words."""
    tokens = [t for t in text.split() if not t.startswith('[slt:') and t != 'spn']
    if not tokens or any(t.startswith('[') for t in tokens):  # [***], [xx***], [overlap], ...
        return None
    return ' '.join(tokens).lower()


def crop(src, segments, dst):
    """Write the concatenation of `segments` (seconds) of `src` to `dst`."""
    audio, sr = sf.read(src)
    pieces = [audio[max(0, int((a - SEGMENT_MARGIN) * sr)):int((b + SEGMENT_MARGIN) * sr)] for a, b in segments]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, np.concatenate(pieces), sr)


def build_ultrasuite(data_root, seed):
    """Clips are restricted to word / sentence / non-word prompts and cropped to the child's speech.

    Child segments come from (in order of preference) the manually revised word labels (UXSSD), the manually
    revised speaker labels, or the automatic speaker labels of the UltraSuite labels release. The cropped audio
    is written to ultrasuite_disorder/child_only/. Transcriptions: UXTD manual transcriptions and UXSSD
    manually revised word labels.
    """
    root = os.path.join(data_root, 'KIDS/ultrasuite_disorder')
    samples, strata, dropped = [], {}, Counter()
    for subset in ('upx', 'uxssd', 'uxtd'):
        prompts = pd.read_csv(os.path.join(root, f'core-{subset}/doc/prompts'), sep='\t')
        prompt_type = dict(zip(prompts['Prompt'].str.strip().str.lower(), prompts['Type']))
        labels = os.path.join(root, 'labels', subset)
        speakers = pd.read_csv(os.path.join(root, f'core-{subset}/doc/speakers'), sep='\t').set_index('speaker_id')
        for spk, info in speakers.iterrows():
            spk_dir = os.path.join(root, f'core-{subset}/core', spk)
            if not os.path.isdir(spk_dir):
                continue
            speaker = f'{subset}/{spk}'  # speaker ids are only unique within a subset
            if subset == 'uxtd':
                sessions = ['']  # UXTD has no session folders
                strata[speaker] = 'typical'
            else:
                sessions = sorted(s for s in os.listdir(spk_dir) if s.startswith('BL'))  # baseline sessions only
                strata[speaker] = info['ssd_subtype'] if subset == 'upx' else 'ssd'
            for session in sessions:
                session_dir = os.path.join(spk_dir, session)
                for name in sorted(f for f in os.listdir(session_dir) if f.endswith('.wav')):
                    wav = os.path.join(session_dir, name)
                    with open(wav.replace('.wav', '.txt')) as f:
                        prompt = f.readline().strip().lower()
                    if prompt_type.get(prompt) not in KEEP_PROMPT_TYPES:
                        dropped['prompt type'] += 1
                        continue

                    label_id = '-'.join(x for x in (spk, session, name[:-4]) if x)  # e.g. 01M-BL1-001A / 07F-001B
                    words = read_lab(os.path.join(labels, f'reference_labels/word-labels/lab/{label_id}.lab'))
                    transcription = None
                    if words:  # manually revised word labels (child's words)
                        segments = [(a, b) for a, b, _ in words]
                        transcription = ' '.join(w for _, _, w in words).lower()
                    else:
                        spk_labels = (read_lab(os.path.join(labels, f'reference_labels/speaker-labels/lab/{label_id}.lab'))
                                      or read_lab(os.path.join(labels, f'speaker_labels/lab/{label_id}.lab')))
                        segments = [(a, b) for a, b, lab in spk_labels if lab == 'CHILD']
                    if not segments:
                        dropped['no child speech'] += 1
                        continue
                    trans_file = os.path.join(labels, f'transcriptions/{label_id}.txt')
                    if subset == 'uxtd' and os.path.exists(trans_file):
                        with open(trans_file) as f:
                            transcription = clean_uxtd_transcription(f.read().strip())

                    child_wav = os.path.join(root, 'child_only', os.path.relpath(wav, root))
                    crop(wav, segments, child_wav)

                    conv = []
                    if transcription:
                        conv += qa('transcribe', f'This is the english transcription, {transcription}')
                    conv += qa('binary_q', random.choice(P['binary_no' if subset == 'uxtd' else 'binary_yes']))
                    if subset == 'upx':
                        conv += qa('multi_q', random.choice(P['multi_a']).format(info['ssd_subtype']))
                    conv += qa('age_q', random.choice(P['age_a']).format(round(float(info['age']))))
                    conv += qa('gender_q', random.choice(P['gender_girl' if info['sex'].lower() == 'female' else 'gender_boy']))
                    samples.append({'audio_name': os.path.relpath(child_wav, data_root), 'conversation': conv,
                                    'speaker': speaker})
    print(f'  ultrasuite: kept {len(samples)} clips, dropped {dict(dropped)}')
    # strata: UPX by SSD subtype, UXSSD as one group, UXTD as one group
    assignment = speaker_split(strata, seed)
    return group(samples, assignment), assignment


# ------------------------- TalkBank ENNI (TD vs. SLI) -------------------------
# The binary disorder question is balanced between TD and SLI clips (see below).
def build_enni(data_root, seed):
    root = os.path.join(data_root, 'KIDS/talkbank_dataset/v1.3/official_v1.3')
    meta = pd.read_csv(os.path.join(root, 'talkbank_childes.csv'))
    meta['id_index'] = meta['audio_file'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    meta = meta.drop_duplicates('id_index').set_index('id_index')

    samples, strata = [], {}
    out_dir = os.path.join(root, 'usable/FASA_ENNI/out')
    for folder_id in sorted(os.listdir(out_dir)):  # one folder = one child
        if folder_id not in meta.index:
            continue
        row = meta.loc[folder_id]
        info = ast.literal_eval(row['metadata'])[0]
        strata[folder_id] = row['group_type']
        for name in sorted(f for f in os.listdir(os.path.join(out_dir, folder_id)) if f.endswith('.mp3')):
            mp3 = os.path.join(out_dir, folder_id, name)
            txt = mp3.replace('.mp3', '.txt')
            conv = []
            if os.path.exists(txt):
                with open(txt) as f:
                    sentence = f.readline().strip()
                conv += qa('what_you_hear', random.choice(P['a_what']).format('child'))
                conv += qa('transcribe', f'This is the english transcription, {sentence}')
            if info.get('age_in_days') is not None:
                conv += qa('age_q', random.choice(P['age_a']).format(int(round(info['age_in_days'] / 365))))
            if info.get('sex') and info['sex'] != '-':
                conv += qa('gender_q', random.choice(P['gender_girl' if info['sex'] == 'female' else 'gender_boy']))
            if row['group_type'] == 'TD':
                conv += qa('binary_q', random.choice(P['binary_no']))
            elif row['group_type'] == 'SLI':
                conv += qa('binary_q', random.choice(P['binary_yes']))
            samples.append({'audio_name': os.path.relpath(mp3, data_root), 'conversation': conv,
                            'speaker': folder_id})
    assignment = speaker_split(strata, seed)

    # TD children produce ~83% of the clips. To balance the binary disorder question, only a random subset
    # of TD children (per split) keeps it, so that TD and SLI clips with this question are roughly 1:1.
    # All clips are kept for the other tasks. The disorder question is the last turn of every ENNI dialogue.
    rng = random.Random(seed)
    clips = Counter(sample['speaker'] for sample in samples)
    for split in SPLITS:
        n_sli = sum(clips[spk] for spk, s in assignment.items() if s == split and strata[spk] == 'SLI')
        td = sorted(spk for spk, s in assignment.items() if s == split and strata[spk] == 'TD')
        rng.shuffle(td)
        keep, n_td = set(), 0
        for spk in td:
            if n_td >= n_sli:
                break
            keep.add(spk)
            n_td += clips[spk]
        for sample in samples:
            if strata[sample['speaker']] == 'TD' and assignment[sample['speaker']] == split and sample['speaker'] not in keep:
                assert sample['conversation'][-2]['value'] in P['binary_q']
                sample['conversation'] = sample['conversation'][:-2]
    return group(samples, assignment), assignment


# ------------------------- English children (native / non-native) -------------------------
def build_english_children(data_root, seed):
    root = os.path.join(data_root, 'KIDS/english_children')
    sources = [('english_free_speech/files_cut_by_sentences', False),
               ('english_words_sentences', True)]  # only studio_mic/sentences for the second one

    samples, strata = [], {}
    for sub, only_studio_sentences in sources:
        for dirpath, _, files in sorted(os.walk(os.path.join(root, sub))):
            if only_studio_sentences and 'studio_mic/sentences' not in dirpath:
                continue
            for name in sorted(f for f in files if f.endswith('.wav')):
                wav = os.path.join(dirpath, name)
                speaker = os.path.relpath(wav, os.path.join(root, sub)).split('/')[0][:2]  # '06_M_native' -> '06'
                if '_M_' in wav:
                    person, gender = 'boy', random.choice(P['gender_boy'])
                elif '_F_' in wav:
                    person, gender = 'girl', random.choice(P['gender_girl'])
                else:
                    person, gender = 'child', random.choice(P['gender_boy'] + P['gender_girl'])
                dialect = 'from UK' if '_native' in wav else 'not from UK' if '_nonNative' in wav else 'unknown'
                strata[speaker] = dialect
                sentence = ' '.join(name.split('.')[0].strip().split('_'))  # transcript is the file name

                conv = qa('what_you_hear', random.choice(P['a_what']).format(person))
                conv += qa('transcribe', f'This is the english transcription, {sentence}')
                conv += qa('dialect', f'The speaker sounds to be {dialect}')
                conv += qa('gender_q', gender)
                samples.append({'audio_name': os.path.relpath(wav, data_root), 'conversation': conv,
                                'speaker': speaker})
    assignment = speaker_split(strata, seed)
    return group(samples, assignment), assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset', help='directory that contains KIDS/')
    parser.add_argument('--output_dir', type=str, default='dataset/json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    builders = {'ultrasuite': build_ultrasuite, 'enni': build_enni, 'english_children': build_english_children}
    merged = {s: [] for s in SPLITS}
    stats = {}
    for name, build in builders.items():
        splits, assignment = build(args.data_root, args.seed)
        stats[name] = {}
        for s in SPLITS:
            with open(os.path.join(args.output_dir, f'{name}_{s}.json'), 'w', encoding='utf-8') as f:
                json.dump(splits[s], f, ensure_ascii=False, indent=2)
            merged[s] += splits[s]
            speakers = sorted(k for k, v in assignment.items() if v == s)
            stats[name][s] = {'samples': len(splits[s]), 'speakers': len(speakers), 'speaker_ids': speakers}
        print(f'{name}: ' + ', '.join(f"{s}={stats[name][s]['samples']} ({stats[name][s]['speakers']} spk)"
                                      for s in SPLITS))

    for s in SPLITS:
        with open(os.path.join(args.output_dir, f'merged_{s}.json'), 'w', encoding='utf-8') as f:
            json.dump(merged[s], f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, 'split_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print('merged: ' + ', '.join(f'{s}={len(v)}' for s, v in merged.items()))


if __name__ == '__main__':
    main()
