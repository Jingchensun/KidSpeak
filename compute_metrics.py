"""Score the JSONL predictions written by inference.py.

Each (question, prediction, answer) triple is routed to a task by matching the question against
the prompt templates in data_prep/prompts, then scored with simple keyword / regex rules:

    task              question templates                  metric
    ----------------  ----------------------------------  -----------------------------------------
    gender            gender_q.txt, whatyouhear.txt       accuracy (boy / girl)
    binary_disorder   disorder_binary_q.txt               accuracy (typical vs. impaired)
    multi_disorder    disorder_multi_q.txt                accuracy (6 SSD subtypes)
    dialect           dialect.txt                         accuracy
    transcription     transctibe_english.txt              WER / CER (%)
    age               age_q.txt                           exact-age accuracy and age-group accuracy

Example:
    python compute_metrics.py --pred outputs/kidspeak_small/ultrasuite.jsonl
"""
import argparse
import json
import os
import re
from collections import defaultdict

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
    correct, total = defaultdict(int), defaultdict(int)
    hyps, refs = [], []

    for r in records:
        q, pred, gt = r['question'], r['prediction'], r['answer']

        if q in prompts['whatyouhear']:
            # "What do you hear?" -> the answer mentions a boy / girl / child
            for k in ('boy', 'girl'):
                if k in pred:
                    total['gender'] += 1
                    correct['gender'] += k in gt

        elif q in prompts['gender']:
            total['gender'] += 1
            correct['gender'] += gender_label(pred) == gender_label(gt)

        elif q in prompts['dialect']:
            total['dialect'] += 1
            key = 'The speaker sounds to be '
            correct['dialect'] += key in pred and key in gt and pred.split(key)[1].strip() == gt.split(key)[1].strip()

        elif q in prompts['binary_disorder']:
            total['binary_disorder'] += 1
            p, g = binary_disorder_label(pred, prompts), binary_disorder_label(gt, prompts)
            correct['binary_disorder'] += p is not None and p == g

        elif q in prompts['multi_disorder']:
            total['multi_disorder'] += 1
            p = disorder_type(pred)
            correct['multi_disorder'] += p is not None and p == disorder_type(gt)

        elif q in prompts['transcription']:
            hyp = strip_prefix(pred, TRANSCRIPTION_PREFIXES)
            ref = strip_prefix(gt, TRANSCRIPTION_PREFIXES[:1])
            if hyp is not None and ref:
                hyps.append(hyp)
                refs.append(ref)

        elif q in prompts['age']:
            p, g = parse_age(pred), parse_age(gt)
            if g is None:
                continue
            total['age'] += 1
            correct['age'] += p == g
            correct['age_group'] += p is not None and age_group(p) is not None and age_group(p) == age_group(g)

    results = {}
    for task in ['gender', 'binary_disorder', 'multi_disorder', 'dialect', 'age']:
        if total[task]:
            results[f'{task}_acc'] = round(100 * correct[task] / total[task], 2)
    if total['age']:
        results['age_group_acc'] = round(100 * correct['age_group'] / total['age'], 2)
    if hyps:
        results['wer'] = round(100 * jiwer.wer(refs, hyps), 2)
        results['cer'] = round(100 * jiwer.cer(refs, hyps), 2)
    results['num_samples'] = {k: v for k, v in total.items() if v}
    if hyps:
        results['num_samples']['transcription'] = len(hyps)
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
