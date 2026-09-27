"""Plot the training loss and the per-task test metrics of every epoch.

    python scripts/plot_curves.py --run outputs/kidspeak_small --log checkpoint/kidspeak_small/train.log

Reads <run>/epoch_<e>/test/<dataset>.metrics.json (and <run>/untrained/test/ as epoch 0, if present) and writes
<run>/figures/{training_loss,task_curves}.png plus <run>/figures/curves.md (all numbers as tables).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = {'ultrasuite': 'UltraSuite', 'enni': 'ENNI', 'english_children': 'English children'}
COLORS = {'ultrasuite': '#2a78d6', 'enni': '#eb6834', 'english_children': '#1baf7a'}  # validated categorical slots 1-3
TASKS = [  # (metric, title, higher is better)
    ('binary_disorder_balanced_acc', 'Disorder (balanced accuracy)', True),
    ('multi_disorder_acc', 'SSD subtype accuracy', True),
    ('age_acc', 'Age accuracy (exact years)', True),
    ('age_group_acc', 'Age-group accuracy', True),
    ('gender_acc', 'Gender accuracy', True),
    ('dialect_acc', 'Native-language accuracy', True),
    ('wer', 'WER (lower is better)', False),
    ('cer', 'CER (lower is better)', False),
]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def style(ax, title, xlabel='epoch'):
    ax.set_title(title, loc='left', fontsize=11, color=INK, fontweight='semibold')
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def load_metrics(run):
    """epoch (0 = before tuning, e + 1 = after epoch e) -> dataset -> metrics."""
    metrics = defaultdict(dict)
    dirs = [(0, os.path.join(run, 'untrained', 'test'))]
    for d in glob.glob(os.path.join(run, 'epoch_*', 'test')):
        dirs.append((int(re.search(r'epoch_(\d+)', d).group(1)) + 1, d))
    for epoch, d in dirs:
        for ds in DATASETS:
            path = os.path.join(d, f'{ds}.metrics.json')
            if os.path.exists(path):
                with open(path) as f:
                    metrics[epoch][ds] = json.load(f)
    return dict(sorted(metrics.items()))


def plot_loss(log, out):
    steps, epochs, losses, accs = [], [], [], []
    with open(log) as f:
        for line in f:
            m = re.search(r'epoch (\d+) step (\d+) loss ([\d.]+) token_acc ([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                steps.append(int(m.group(2)))
                losses.append(float(m.group(3)))
                accs.append(float(m.group(4)))
    steps_per_epoch = (max(steps) + 1) / (max(epochs) + 1)
    x = [s / steps_per_epoch for s in steps]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=150)
    for ax, values, title in [(axes[0], losses, 'Training loss'), (axes[1], accs, 'Training token accuracy (%)')]:
        ax.plot(x, values, color='#86b6ef', linewidth=0.8, label='every 5 steps')
        per_epoch = defaultdict(list)
        for e, v in zip(epochs, values):
            per_epoch[e].append(v)
        ex = [e + 0.5 for e in sorted(per_epoch)]  # the mean over an epoch sits in its middle
        ey = [sum(per_epoch[e]) / len(per_epoch[e]) for e in sorted(per_epoch)]
        ax.plot(ex, ey, color='#184f95', linewidth=2, marker='o', markersize=4, label='epoch mean')
        ax.annotate(f'{ey[-1]:.2f}', (ex[-1], ey[-1]), textcoords='offset points', xytext=(6, 0),
                    fontsize=8, color=INK2, va='center')
        style(ax, title)
        ax.set_xticks(range(0, int(max(ex) + 0.5) + 1))
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_tasks(metrics, out):
    epochs = list(metrics)
    tasks = [t for t in TASKS if any(t[0] in m for e in epochs for m in metrics[e].values())]
    cols = 4
    rows = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), dpi=150, squeeze=False)
    for ax, (key, title, _) in zip(axes.flat, tasks):
        ends = []
        for ds, name in DATASETS.items():
            xs = [e for e in epochs if key in metrics[e].get(ds, {})]
            if not xs:
                continue
            ys = [metrics[e][ds][key] for e in xs]
            ax.plot(xs, ys, color=COLORS[ds], linewidth=2, marker='o', markersize=4, label=name)
            ends.append((xs[-1], ys[-1]))
            base_key = key.replace('_acc', '_majority_baseline')
            base = next((metrics[e][ds].get(base_key) for e in xs if base_key in metrics[e][ds]), None)
            if base is not None and key not in ('binary_disorder_balanced_acc', 'wer', 'cer'):
                ax.axhline(base, color=COLORS[ds], linewidth=1, linestyle='--', alpha=0.6)
        if key == 'binary_disorder_balanced_acc':  # 50% for every dataset
            ax.axhline(50, color=INK2, linewidth=1, linestyle='--', alpha=0.6)
        # end-of-line values, pushed apart so that they do not overlap
        lo, hi = ax.get_ylim()
        gap, placed = (hi - lo) * 0.06, []
        for x, y in sorted(ends, key=lambda e: e[1]):
            ty = max(y, placed[-1] + gap) if placed else y
            placed.append(ty)
            ax.annotate(f'{y:.1f}', (x, y), xytext=(x + 0.25, ty), textcoords='data', fontsize=8, color=INK2, va='center',
                        ha='left', annotation_clip=False)
        style(ax, title)
        ax.set_xticks(epochs)
        ax.set_xlim(min(epochs) - 0.3, max(epochs) + 1.2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis('off')
    fig.text(0.01, 0.005, 'Epoch 0 = before instruction tuning. Dashed lines: majority-class baseline of the dataset '
             '(50% for balanced accuracy).', fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def write_tables(metrics, out):
    lines = ['Epoch 0 = before instruction tuning. All numbers in %.', '']
    for ds, name in DATASETS.items():
        keys = [(k, t) for k, t, _ in TASKS if any(k in metrics[e].get(ds, {}) for e in metrics)]
        lines += [f'**{name}**', '', '| epoch | ' + ' | '.join(t for _, t in keys) + ' |',
                  '|---:|' + '---:|' * len(keys)]
        for e in metrics:
            if ds in metrics[e]:
                lines.append(f'| {e} | ' + ' | '.join(f'{metrics[e][ds][k]:.1f}' if k in metrics[e][ds] else '–'
                                                      for k, _ in keys) + ' |')
        lines.append('')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='outputs/kidspeak_small')
    parser.add_argument('--log', default='checkpoint/kidspeak_small/train.log')
    args = parser.parse_args()
    fig_dir = os.path.join(args.run, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    if os.path.exists(args.log):
        plot_loss(args.log, os.path.join(fig_dir, 'training_loss.png'))
    metrics = load_metrics(args.run)
    plot_tasks(metrics, os.path.join(fig_dir, 'task_curves.png'))
    write_tables(metrics, os.path.join(fig_dir, 'curves.md'))
    print(f'[!] Wrote figures and tables to {fig_dir} (epochs: {list(metrics)})')


if __name__ == '__main__':
    main()
