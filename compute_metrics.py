"""Score the JSONL predictions written by inference.py.

Each (question, prediction, answer) triple is routed to a task by matching the question against
the prompt templates in data_prep/prompts, then scored with simple keyword / regex rules:

    task              question templates                  metric
    ----------------  ----------------------------------  -----------------------------------------
    gender            gender_q.txt                        accuracy (boy / girl)
    binary_disorder   disorder_binary_q.txt               accuracy (typical vs. impaired)
    multi_disorder    disorder_multi_q.txt                accuracy (6 SSD subtypes)
    dialect           dialect.txt                         accuracy
    transcription     transctibe_english.txt              WER / CER (%)
    age               age_q.txt                           exact-age accuracy and age-group accuracy

Every classification task also reports its majority-class baseline; the binary disorder task also reports
the balanced accuracy.

Example:
    python compute_metrics.py --pred outputs/kidspeak_small/ultrasuite.jsonl
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import jiwer

DISORDER_TYPES = [  # longest first, so that "phonological disorder" does not shadow its superstrings
    'inconsistent phonological disorder',
    'childhood apraxia of speech',
    'articulation disorder',
    'phonological disorder',
    'phonological delay',
    'vowel disorder',
]
NUMBER_WORDS = {w: i for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
    'sixteen seventeen eighteen nineteen'.split())}
TRANSCRIPTION_PREFIXES = ['This is the english transcription, ', 'This is the english transcription ']


def load_prompts(prompt_dir):
    def read(name):
        with open(os.path.join(prompt_dir, name)) as f:
            return [line.strip() for line in f]
    return {
        'whatyouhear': read('whatyouhear.txt'),
        'gender': read('gender_q.txt'),
        'dialect': read('dialect.txt'),
        'binary_disorder': read('disorder_binary_q.txt'),
        'binary_disorder_yes': read('binary_disorder_binary_isdis.txt'),
        'binary_disorder_no': read('binary_disorder_binary_nodis.txt'),
        'multi_disorder': read('disorder_multi_q.txt'),
        'transcription': read('transctibe_english.txt'),
        'age': read('age_q.txt'),
    }


def gender_label(text):
    label = None
    if 'boy' in text or 'male' in text:
        label = 'boy'
    if 'girl' in text or 'female' in text:
        label = 'girl'
    return label


def binary_disorder_label(text, prompts):
    if any(p in text for p in prompts['binary_disorder_yes']):
        return 'disorder'
    if any(p in text for p in prompts['binary_disorder_no']):
        return 'typical'
    return None


def disorder_type(text):
    return next((d for d in DISORDER_TYPES if d in text), None)


def parse_age(text):
    digits = ''.join(re.findall(r'\d+', text))
    if digits:
        return int(digits)
    words = [NUMBER_WORDS[w] for w in text.lower().split() if w in NUMBER_WORDS]
    return sum(words) if words else None


def age_group(age):
    for upper, name in [(3, 'toddler'), (5, 'preschool'), (8, 'early school age'),
                        (12, 'later school age'), (17, 'teenager')]:
        if 0 <= age <= upper:
            return name
    return None


def strip_prefix(text, prefixes):
    for p in prefixes:
        if p in text:
            return text.split(p, 1)[1].strip()
    return None


def compute_metrics(records, prompts):
    """Accuracy per classification task, WER / CER for transcription.

    For every classification task the majority-class baseline (always answering the most frequent reference
    label) is reported next to the accuracy, and the binary disorder task also gets the balanced accuracy
    (mean recall of the two classes).
    """
    pairs = defaultdict(list)  # task -> [(predicted label, reference label)]
    hyps, refs = [], []

    for r in records:
        q, pred, gt = r['question'], r['prediction'], r['answer']
        if q in prompts['gender']:
            pairs['gender'].append((gender_label(pred), gender_label(gt)))
        elif q in prompts['dialect']:
            key = 'The speaker sounds to be '
            pairs['dialect'].append((pred.split(key)[1].strip() if key in pred else None, gt.split(key)[1].strip()))
        elif q in prompts['binary_disorder']:
            pairs['binary_disorder'].append((binary_disorder_label(pred, prompts), binary_disorder_label(gt, prompts)))
        elif q in prompts['multi_disorder']:
            pairs['multi_disorder'].append((disorder_type(pred), disorder_type(gt)))
        elif q in prompts['age']:
            p, g = parse_age(pred), parse_age(gt)
            if g is not None:
                pairs['age'].append((p, g))
                pairs['age_group'].append((age_group(p) if p is not None else None, age_group(g)))
        elif q in prompts['transcription']:
            hyp = strip_prefix(pred, TRANSCRIPTION_PREFIXES)
            ref = strip_prefix(gt, TRANSCRIPTION_PREFIXES[:1])
            if ref:
                hyps.append(hyp or '')
                refs.append(ref)

    results, num_samples = {}, {}
    for task in ['gender', 'binary_disorder', 'multi_disorder', 'dialect', 'age', 'age_group']:
        if not pairs[task]:
            continue
        ref_counts = Counter(g for _, g in pairs[task])
        results[f'{task}_acc'] = round(100 * sum(p == g for p, g in pairs[task]) / len(pairs[task]), 2)
        results[f'{task}_majority_baseline'] = round(100 * ref_counts.most_common(1)[0][1] / len(pairs[task]), 2)
        num_samples[task] = len(pairs[task])
    if pairs['binary_disorder']:
        recalls = [sum(p == g for p, g in pairs['binary_disorder'] if g == c) / n
                   for c, n in Counter(g for _, g in pairs['binary_disorder']).items()]
        results['binary_disorder_balanced_acc'] = round(100 * sum(recalls) / len(recalls), 2)
    if hyps:
        results['wer'] = round(100 * jiwer.wer(refs, hyps), 2)
        results['cer'] = round(100 * jiwer.cer(refs, hyps), 2)
        num_samples['transcription'] = len(hyps)
    results['num_samples'] = num_samples
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='JSONL file written by inference.py')
    parser.add_argument('--prompt_dir', type=str, default='data_prep/prompts')
    parser.add_argument('--output', type=str, default=None, help='defaults to <pred>.metrics.json')
    args = parser.parse_args()

    with open(args.pred, encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    results = compute_metrics(records, load_prompts(args.prompt_dir))

    output = args.output or os.path.splitext(args.pred)[0] + '.metrics.json'
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f'[!] Saved to {output}')


if __name__ == '__main__':
    main()
