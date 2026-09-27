"""Collect <dataset>.metrics.json from several runs into one Markdown table per dataset.

The first row is the majority-class baseline (always answering the most frequent test label; 50% balanced
accuracy for the binary task; not defined for transcription), taken from the metrics of the first run.

    python scripts/summarize_results.py \
        "Before tuning=outputs/kidspeak_small/untrained/test" \
        "Instruction-tuned=outputs/kidspeak_small/epoch_0/test"
"""
import json
import os
import sys

DATASETS = {'ultrasuite': 'UltraSuite', 'enni': 'ENNI', 'english_children': 'English children'}
COLUMNS = [  # (metric key, header); a column is shown when any run of the dataset has it
    ('binary_disorder_acc', 'Disorder acc'), ('binary_disorder_balanced_acc', 'Disorder bal. acc'),
    ('multi_disorder_acc', 'SSD subtype acc'), ('age_acc', 'Age acc'), ('age_group_acc', 'Age-group acc'),
    ('gender_acc', 'Gender acc'), ('dialect_acc', 'Native-lang. acc'), ('wer', 'WER ↓'), ('cer', 'CER ↓'),
]


def main():
    runs = [arg.split('=', 1) for arg in sys.argv[1:]]
    for ds, title in DATASETS.items():
        results = []
        for name, path in runs:
            with open(os.path.join(path, f'{ds}.metrics.json')) as f:
                results.append((name, json.load(f)))
        majority = {k[:-len('_majority_baseline')] + '_acc': v for k, v in results[0][1].items()
                    if k.endswith('_majority_baseline')}
        if 'binary_disorder_acc' in majority:
            majority['binary_disorder_balanced_acc'] = 50.0
        results.insert(0, ('Majority class', majority))
        cols = [(k, h) for k, h in COLUMNS if any(k in r for _, r in results)]
        n = results[-1][1]['num_samples']
        print(f'**{title}** (test: ' + ', '.join(f'{k.replace("_", " ")} {v:,}' for k, v in n.items()) + ')\n')
        print('| | ' + ' | '.join(h for _, h in cols) + ' |')
        print('|---|' + '---:|' * len(cols))
        for name, r in results:
            print(f'| {name} | ' + ' | '.join(f'{r[k]:.1f}' if k in r else '–' for k, _ in cols) + ' |')
        print()


if __name__ == '__main__':
    main()
