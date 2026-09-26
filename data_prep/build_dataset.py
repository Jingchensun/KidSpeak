"""Build the KidSpeak instruction-tuning JSON files from the raw corpora.

Expected layout under --data_root (audio_name paths in the JSON are relative to it):

    dataset/KIDS/
    ├── ultrasuite_disorder/
    │   └── core-upx/                      # official UltraSuite download
    │       ├── doc/speakers               # speaker metadata incl. SSD subtype (TSV)
    │       └── core/<speaker>/BL*/*.wav (+ .txt)
    ├── talkbank_dataset/v1.3/official_v1.3/
    │   ├── talkbank_childes.csv           # FASA metadata
    │   └── usable/FASA_ENNI/out/<id>/*.mp3 (+ .txt)
    └── english_children/                  # english_children.zip from https://zenodo.org/records/200495
        ├── english_free_speech/files_cut_by_sentences/**.wav
        └── english_words_sentences/<speaker>/**/studio_mic/sentences/*.wav

Outputs (in --output_dir): {ultrasuite_disorder,talkbank_v1_3_enni_post,english_children}_{train,val,test}.json
and merged_{train,val,test}.json.

Note: prompts are sampled randomly, so a rebuild is not byte-identical to the released JSON files.
Use the released files (see README) to reproduce the paper numbers.
"""
import argparse
import ast
import json
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')


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


def split(samples, seeds=(42, 42)):
    """70 / 20 / 10 random split."""
    train, rest = train_test_split(samples, test_size=0.3, random_state=seeds[0])
    val, test = train_test_split(rest, test_size=1 / 3, random_state=seeds[1])
    return {'train': train, 'val': val, 'test': test}


# ------------------------- UltraSuite (speech sound disorders) -------------------------
def build_ultrasuite(data_root):
    root = os.path.join(data_root, 'KIDS/ultrasuite_disorder')
    speakers = pd.read_csv(os.path.join(root, 'core-upx/doc/speakers'), sep='\t')
    subtype = dict(zip(speakers['speaker_id'], speakers['ssd_subtype']))
    age = dict(zip(speakers['speaker_id'], speakers['age']))
    sex = dict(zip(speakers['speaker_id'], speakers['sex']))

    samples = []
    for dirpath, _, files in sorted(os.walk(os.path.join(root, 'core-upx/core'))):
        session, speaker = dirpath.split('/')[-1], dirpath.split('/')[-2]
        if 'BL' not in session or speaker not in subtype:  # baseline sessions only
            continue
        for name in sorted(f for f in files if f.endswith('.wav')):
            wav = os.path.join(dirpath, name)
            if not os.path.exists(wav.replace('.wav', '.txt')):
                continue
            conv = qa('binary_q', random.choice(P['binary_yes']))
            conv += qa('multi_q', random.choice(P['multi_a']).format(subtype[speaker]))
            if speaker in age:
                conv += qa('age_q', random.choice(P['age_a']).format(round(float(age[speaker]))))
            if speaker in sex:
                conv += qa('gender_q', random.choice(P['gender_girl' if sex[speaker].lower() == 'female' else 'gender_boy']))
            samples.append({'audio_name': os.path.relpath(wav, data_root), 'conversation': conv})
    return split(samples)


# ------------------------- TalkBank ENNI (TD vs. SLI) -------------------------
def build_enni(data_root):
    root = os.path.join(data_root, 'KIDS/talkbank_dataset/v1.3/official_v1.3')
    meta = pd.read_csv(os.path.join(root, 'talkbank_childes.csv'))
    meta['id_index'] = meta['audio_file'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    meta = meta.drop_duplicates('id_index').set_index('id_index')

    samples = []
    for dirpath, _, files in sorted(os.walk(os.path.join(root, 'usable/FASA_ENNI/out'))):
        folder_id = os.path.basename(dirpath)
        if folder_id not in meta.index:
            continue
        row = meta.loc[folder_id]
        info = ast.literal_eval(row['metadata'])[0]
        for name in sorted(f for f in files if f.endswith('.mp3')):
            mp3 = os.path.join(dirpath, name)
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
            samples.append({'audio_name': os.path.relpath(mp3, data_root), 'conversation': conv})
    return split(samples)


# ------------------------- English children (native / non-native) -------------------------
def build_english_children(data_root):
    root = os.path.join(data_root, 'KIDS/english_children')
    sources = [('english_free_speech/files_cut_by_sentences', False),
               ('english_words_sentences', True)]  # only studio_mic/sentences for the second one

    samples = []
    for sub, only_studio_sentences in sources:
        for dirpath, _, files in sorted(os.walk(os.path.join(root, sub))):
            if only_studio_sentences and 'studio_mic/sentences' not in dirpath:
                continue
            for name in sorted(f for f in files if f.endswith('.wav')):
                wav = os.path.join(dirpath, name)
                if '_M_' in wav:
                    person, gender = 'boy', random.choice(P['gender_boy'])
                elif '_F_' in wav:
                    person, gender = 'girl', random.choice(P['gender_girl'])
                else:
                    person, gender = 'child', random.choice(P['gender_boy'] + P['gender_girl'])
                dialect = 'from UK' if '_native' in wav else 'not from UK' if '_nonNative' in wav else 'unknown'
                sentence = ' '.join(name.split('.')[0].strip().split('_'))  # transcript is the file name

                conv = qa('what_you_hear', random.choice(P['a_what']).format(person))
                conv += qa('transcribe', f'This is the english transcription, {sentence}')
                conv += qa('dialect', f'The speaker sounds to be {dialect}')
                conv += qa('gender_q', gender)
                samples.append({'audio_name': os.path.relpath(wav, data_root), 'conversation': conv})
    return split(samples, seeds=(52, 62))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset', help='directory that contains KIDS/')
    parser.add_argument('--output_dir', type=str, default='dataset/json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    builders = {
        'ultrasuite_disorder': build_ultrasuite,
        'talkbank_v1_3_enni_post': build_enni,
        'english_children': build_english_children,
    }
    merged = {'train': [], 'val': [], 'test': []}
    for name, build in builders.items():
        splits = build(args.data_root)
        for s, samples in splits.items():
            with open(os.path.join(args.output_dir, f'{name}_{s}.json'), 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            merged[s] += samples
        print(f'{name}: ' + ', '.join(f'{s}={len(v)}' for s, v in splits.items()))

    for s, samples in merged.items():
        with open(os.path.join(args.output_dir, f'merged_{s}.json'), 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    print('merged: ' + ', '.join(f'{s}={len(v)}' for s, v in merged.items()))


if __name__ == '__main__':
    main()
