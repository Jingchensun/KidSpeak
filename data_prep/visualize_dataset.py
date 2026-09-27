"""Render an HTML overview of the built datasets: split statistics, label distributions and audio examples.

    python data_prep/visualize_dataset.py \
        --json_dir dataset/json --audio_root dataset \
        --pred_dir outputs/kidspeak_small/epoch_0/test      # optional: show model answers on test examples

Writes dataset_overview.html and copies the example clips to demos/ (both next to each other, so the page can
play the audio when opened from disk or served as a static site).
"""
import argparse
import html
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from compute_metrics import (TRANSCRIPTION_PREFIXES, binary_disorder_label, disorder_type,  # noqa: E402
                             gender_label, load_prompts, parse_age, strip_prefix)

DATASETS = {
    'ultrasuite': {
        'title': 'UltraSuite',
        'speech': 'Prompted words, sentences, non-words',
        'about': 'Speech-therapy recordings. UPX (20 children with speech sound disorders, SSD subtype labelled), '
                 'UXSSD (8 children with SSD) and UXTD (58 typically developing children). Clips are cropped to the '
                 "child's speech; the therapist's voice is removed.",
        'license': 'CC BY-NC 4.0',
        'link': 'https://ultrasuite.github.io/',
    },
    'enni': {
        'title': 'ENNI',
        'speech': 'Story telling (spontaneous)',
        'about': 'Edmonton Narrative Norms Instrument (TalkBank CHILDES): children tell stories from picture books. '
                 'One recording per child, segmented into utterances with FASA. Typically developing (TD) children '
                 'and children with a specific language impairment (SLI).',
        'license': 'CC BY-NC-SA 3.0',
        'link': 'https://talkbank.org/childes/access/Clinical-Eng/ENNI.html',
    },
    'english_children': {
        'title': 'English children',
        'speech': 'Read sentences, story retelling',
        'about': 'Kennedy et al. (HRI 2017): UK primary-school children, native and non-native English speakers. '
                 'The transcription is the file name.',
        'license': 'CC BY 4.0',
        'link': 'https://zenodo.org/records/200495',
    },
}
TASKS = [('transcription', 'Transcription'), ('binary_disorder', 'Disorder'), ('multi_disorder', 'SSD subtype'),
         ('age', 'Age'), ('gender', 'Gender'), ('dialect', 'Native language'), ('whatyouhear', 'Description')]
SPLITS = ('train', 'test')
DURATION_BINS = [(0, 1, '<1 s'), (1, 2, '1-2 s'), (2, 4, '2-4 s'), (4, 8, '4-8 s'), (8, 16, '8-16 s'), (16, 1e9, '>16 s')]
TRANSCRIPTION = 'This is the english transcription, '


def task_of(question, prompts):
    return next((key for key, _ in TASKS if question in prompts[key]), None)


def subset_of(ds, audio_name):
    if ds == 'ultrasuite':
        return audio_name.split('/')[3].replace('core-', '').upper()  # UPX / UXSSD / UXTD
    if ds == 'english_children':
        return 'read sentences' if 'english_words_sentences' in audio_name else 'story retelling'
    return None


def speaker_of(ds, audio_name):
    parts = audio_name.split('/')
    if ds == 'ultrasuite':
        return f"{parts[3].replace('core-', '')}/{parts[5]}"
    if ds == 'enni':
        return parts[-2]
    base = 'english_words_sentences' if 'english_words_sentences' in audio_name else 'files_cut_by_sentences'
    return parts[parts.index(base) + 1]


def collect(json_dir, audio_root, prompts):
    """Per dataset and split: number of clips / children / hours and label counters."""
    data, stats = {}, {}
    for ds in DATASETS:
        stats[ds] = {}
        for split in SPLITS:
            with open(os.path.join(json_dir, f'{ds}_{split}.json')) as f:
                samples = json.load(f)
            data[(ds, split)] = samples
            st = defaultdict(Counter)
            hours, speakers = 0.0, set()
            for s in samples:
                speakers.add(speaker_of(ds, s['audio_name']))
                if subset_of(ds, s['audio_name']):
                    st['subset'][subset_of(ds, s['audio_name'])] += 1
                duration = sf.info(os.path.join(audio_root, s['audio_name'])).duration
                hours += duration / 3600
                st['duration'][next(label for lo, hi, label in DURATION_BINS if lo <= duration < hi)] += 1
                conv = s['conversation']
                for i in range(0, len(conv), 2):
                    q, a = conv[i]['value'], conv[i + 1]['value']
                    task = task_of(q, prompts)
                    st['tasks'][task] += 1
                    if task == 'binary_disorder':
                        st['binary'][binary_disorder_label(a, prompts)] += 1
                    elif task == 'multi_disorder':
                        st['subtype'][disorder_type(a)] += 1
                    elif task == 'gender':
                        st['gender'][gender_label(a)] += 1
                    elif task == 'age':
                        st['age'][str(parse_age(a))] += 1
                    elif task == 'dialect':
                        st['dialect'][a.split('to be ')[-1]] += 1
            stats[ds][split] = {'clips': len(samples), 'speakers': len(speakers), 'hours': hours,
                                'dist': {k: dict(v) for k, v in st.items()}}
    return data, stats


def pick_examples(ds, data, seed, prompts):
    """A few diverse, deterministic examples per split."""
    rng = random.Random(seed)

    def asks(sample, task):
        return any(t['value'] in prompts[task] for t in sample['conversation'][::2])

    def binary_answer(sample):
        conv = sample['conversation']
        return next((binary_disorder_label(conv[i + 1]['value'], prompts) for i in range(0, len(conv), 2)
                     if conv[i]['value'] in prompts['binary_disorder']), None)

    picks = []
    for split in SPLITS:
        samples = data[(ds, split)]
        if ds == 'ultrasuite':
            for subset, task in [('UXTD', 'transcription'), ('UPX', 'multi_disorder'), ('UXSSD', 'transcription')]:
                pool = [s for s in samples if subset_of(ds, s['audio_name']) == subset and asks(s, task)]
                pool = [s for s in pool if asks(s, 'binary_disorder')] or pool
                if pool:
                    picks.append((split, rng.choice(pool)))
        elif ds == 'enni':
            for label in ('typical', 'disorder'):
                pool = [s for s in samples if binary_answer(s) == label
                        and 4 <= len(strip_prefix(s['conversation'][3]['value'], TRANSCRIPTION_PREFIXES[:1]).split()) <= 14]
                if pool:
                    picks.append((split, rng.choice(pool)))
        else:
            for kind in ('story retelling', 'read sentences'):
                pool = [s for s in samples if subset_of(ds, s['audio_name']) == kind]
                if pool:
                    picks.append((split, rng.choice(pool)))
    return picks


def load_predictions(pred_dir):
    preds = {}
    if not pred_dir or not os.path.isdir(pred_dir):
        return preds
    for fn in os.listdir(pred_dir):
        if fn.endswith('.jsonl'):
            with open(os.path.join(pred_dir, fn)) as f:
                for line in f:
                    r = json.loads(line)
                    preds[(r['audio'], r['question'])] = r['prediction']
    return preds


def example_card(ds, split, idx, sample, audio_root, out_dir, preds, prompts):
    src = os.path.join(audio_root, sample['audio_name'])
    ext = os.path.splitext(src)[1]
    rel = f'demos/{ds}/{split}_{idx}{ext}'
    os.makedirs(os.path.join(out_dir, os.path.dirname(rel)), exist_ok=True)
    shutil.copy(src, os.path.join(out_dir, rel))
    players = [('', rel)]
    if ds == 'ultrasuite':  # also offer the original recording with the therapist's voice
        orig_rel = f'demos/{ds}/{split}_{idx}_original{ext}'
        shutil.copy(src.replace('/child_only/', '/'), os.path.join(out_dir, orig_rel))
        players = [('child only', rel), ('original', orig_rel)]

    rows = []
    conv = sample['conversation']
    for i in range(0, len(conv), 2):
        q, a = conv[i]['value'], conv[i + 1]['value']
        pred = preds.get((sample['audio_name'], q)) if split == 'test' else None
        pred_html = f'<div class="pred">model: {html.escape(pred.replace(TRANSCRIPTION, ""))}</div>' if pred else ''
        rows.append(f'<div class="k">{dict(TASKS).get(task_of(q, prompts), "")}</div>'
                    f'<div class="v"><div class="q">{html.escape(q)}</div>{html.escape(a.replace(TRANSCRIPTION, ""))}'
                    f'{pred_html}</div>')
    audio = ''.join(f'<label class="player">{f"<span>{lab}</span>" if lab else ""}'
                    f'<audio controls preload="none" src="{p}"></audio></label>' for lab, p in players)
    tags = ([subset_of(ds, sample['audio_name'])] if subset_of(ds, sample['audio_name']) else []) + \
        [f'{sf.info(src).duration:.1f} s']
    return (f'<article class="ex"><div class="ex-head" title="{html.escape(sample["audio_name"])}">'
            f'<b>child {html.escape(speaker_of(ds, sample["audio_name"]))}</b>'
            f'{"".join(f"<span>{t}</span>" for t in tags)}</div>{audio}<div class="qa">{"".join(rows)}</div></article>')


def chart_specs(ds, stats):
    """(key, title, category order) of the distribution charts of a dataset."""
    tr, te = stats[ds]['train']['dist'], stats[ds]['test']['dist']

    def cats(key, order=None):
        keys = set(tr.get(key, {})) | set(te.get(key, {}))
        if order:
            return [k for k in order if k in keys]
        return sorted(keys, key=lambda k: -(tr.get(key, {}).get(k, 0) + te.get(key, {}).get(k, 0)))

    specs = []
    if 'subset' in tr:
        specs.append(('subset', 'Subset' if ds == 'ultrasuite' else 'Recording type', cats('subset')))
    if 'binary' in tr:
        specs.append(('binary', 'Disorder question', cats('binary', ['typical', 'disorder'])))
    if 'subtype' in tr:
        specs.append(('subtype', 'SSD subtype', cats('subtype')))
    if 'age' in tr:
        specs.append(('age', 'Age (years)', sorted(set(tr['age']) | set(te['age']), key=int)))
    if 'gender' in tr:
        specs.append(('gender', 'Gender', cats('gender', ['boy', 'girl'])))
    if 'dialect' in tr:
        specs.append(('dialect', 'Native language', cats('dialect')))
    specs.append(('duration', 'Clip duration', cats('duration', [b[2] for b in DURATION_BINS])))
    return specs


def render(stats, examples, out_path, artifact=False):
    def fmt(n):
        return f'{n:,}'

    rows = []
    for ds, meta in DATASETS.items():
        st = stats[ds]
        tasks = ''.join(f'<span class="tag">{name}</span>' for key, name in TASKS
                        if key != 'whatyouhear' and st['train']['dist']['tasks'].get(key))
        rows.append(f'<tr><th><a href="#{ds}" onclick="show(\'{ds}\')">{meta["title"]}</a></th>'
                    f'<td>{fmt(st["train"]["speakers"])} / {fmt(st["test"]["speakers"])}</td>'
                    f'<td>{fmt(st["train"]["clips"])} / {fmt(st["test"]["clips"])}</td>'
                    f'<td>{st["train"]["hours"] + st["test"]["hours"]:.1f}</td>'
                    f'<td class="l">{meta["speech"]}</td><td class="l">{tasks}</td></tr>')
    tot = {s: {k: sum(stats[ds][s][k] for ds in DATASETS) for k in ('clips', 'speakers', 'hours')} for s in SPLITS}
    rows.append(f'<tr class="total"><th>Total</th><td>{fmt(tot["train"]["speakers"])} / {fmt(tot["test"]["speakers"])}</td>'
                f'<td>{fmt(tot["train"]["clips"])} / {fmt(tot["test"]["clips"])}</td>'
                f'<td>{tot["train"]["hours"] + tot["test"]["hours"]:.1f}</td><td></td><td></td></tr>')

    tabs, panels = [], []
    for ds, meta in DATASETS.items():
        st = stats[ds]
        tabs.append(f'<button role="tab" data-ds="{ds}" onclick="show(\'{ds}\')">{meta["title"]}</button>')
        split_line = ' &nbsp;&middot;&nbsp; '.join(
            f'<span class="dot {s}"></span><b>{s}</b> {fmt(st[s]["clips"])} clips, {st[s]["speakers"]} children'
            for s in SPLITS)
        charts = ''.join(f'<figure class="chart" data-ds="{ds}" data-key="{key}" data-cats=\'{html.escape(json.dumps(c))}\'>'
                         f'<figcaption>{title}</figcaption></figure>' for key, title, c in chart_specs(ds, stats))
        cols = ''.join(f'<div class="col"><h4><span class="dot {s}"></span>{s}</h4>{"".join(examples[ds][s])}</div>'
                       for s in SPLITS)
        panels.append(
            f'<section class="panel" id="{ds}" role="tabpanel">'
            f'<p class="about">{html.escape(meta["about"])} <a href="{meta["link"]}">Source</a> &middot; {meta["license"]}</p>'
            f'<p class="splits">{split_line}</p>'
            f'<h3>Label distributions <span class="hint">left: train, right: test &middot; hover a slice for counts</span></h3>'
            f'<div class="charts">{charts}</div>'
            f'<h3>Examples <label class="toggle"><input type="checkbox" '
            f'onchange="document.body.classList.toggle(\'show-q\', this.checked)"> show questions</label></h3>'
            f'<div class="cols">{cols}</div></section>')

    page = (TEMPLATE.replace('%ROWS%', ''.join(rows)).replace('%TABS%', ''.join(tabs))
            .replace('%PANELS%', ''.join(panels)).replace('%STATS%', json.dumps(stats)))
    if artifact:  # Claude Artifact: no document skeleton, the viewer controls the theme
        page = page.replace('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                            '<meta name="viewport" content="width=device-width, initial-scale=1">\n', '')
        page = page.replace('</head>\n<body>\n', '').replace('</body>\n</html>\n', '')
        page = page.replace('<button class="theme" onclick="toggleTheme()">Light / dark</button>\n', '')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(page)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KidSpeak Datasets</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {
  color-scheme: light;
  --bg: #fcfcfb; --panel: #ffffff; --line: #e6e5e0; --grid: #eeede9;
  --ink: #111110; --ink-2: #52514e; --ink-3: #85847d;
  --train: #2a78d6; --test: #eb6834; --soft: #f3f2ee; --pred: #fdf0ea; --link: #1c5cab;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #008300; --c7: #4a3aa7; --c8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg: #1a1a19; --panel: #222221; --line: #363633; --grid: #2d2d2b;
    --ink: #f5f5f2; --ink-2: #c3c2b7; --ink-3: #8f8e86;
    --train: #3987e5; --test: #d95926; --soft: #2b2b29; --pred: #36241c; --link: #86b6ef;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1a1a19; --panel: #222221; --line: #363633; --grid: #2d2d2b;
  --ink: #f5f5f2; --ink-2: #c3c2b7; --ink-3: #8f8e86;
  --train: #3987e5; --test: #d95926; --soft: #2b2b29; --pred: #36241c; --link: #86b6ef;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
.leg td, .ex-head span, .tag { font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; }
h1, h3 { text-wrap: balance; }
main { max-width: 1120px; margin: 0 auto; padding: 40px 16px 80px; }
a { color: var(--link); text-decoration: none; }
h1 { font-size: 26px; margin: 0; letter-spacing: -.01em; }
.sub { color: var(--ink-2); margin: 6px 0 0; max-width: 760px; }
.theme { float: right; border: 1px solid var(--line); background: var(--panel); color: var(--ink-2);
  border-radius: 6px; padding: 4px 10px; font: inherit; font-size: 13px; cursor: pointer; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-3); margin: 36px 0 10px; font-weight: 600; }
h3 { font-size: 15px; margin: 28px 0 10px; }
.hint { font-weight: 400; color: var(--ink-3); font-size: 13px; margin-left: 6px; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
table th:first-child, table td.l { font-family: "IBM Plex Sans", -apple-system, "Segoe UI", sans-serif; font-size: 14px; }
th, td { padding: 10px 14px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th, td.l { text-align: left; }
td.l { white-space: normal; min-width: 150px; }
thead th { color: var(--ink-3); font-weight: 500; font-size: 12px; line-height: 1.3; vertical-align: bottom; }
thead th.r { text-align: right; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: 0; }
tr.total { color: var(--ink-2); }
.tag { display: inline-block; font-size: 12px; background: var(--soft); border-radius: 4px; padding: 0 6px; margin: 1px 4px 1px 0; }
ul.notes { color: var(--ink-2); padding-left: 18px; margin: 14px 0 0; font-size: 14px; }
ul.notes li { margin: 3px 0; }
[role=tablist] { display: flex; gap: 4px; border-bottom: 1px solid var(--line); margin-top: 40px; overflow-x: auto; }
[role=tab] { font: inherit; font-weight: 600; background: none; border: 0; color: var(--ink-3); padding: 10px 14px;
  border-bottom: 2px solid transparent; margin-bottom: -1px; cursor: pointer; white-space: nowrap; }
[role=tab][aria-selected=true] { color: var(--ink); border-bottom-color: var(--ink); }
.panel { display: none; } .panel.on { display: block; }
.about { color: var(--ink-2); margin: 18px 0 6px; max-width: 820px; }
.splits { margin: 0; font-size: 14px; color: var(--ink-2); }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }
.dot.train { background: var(--train); } .dot.test { background: var(--test); }
.charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; align-items: start; }
.chart { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px 8px; }
.chart figcaption { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.pies { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 6px 0 8px; }
.pie { text-align: center; }
.pie svg { display: block; width: 100%; max-width: 132px; height: auto; margin: 0 auto; }
.pie svg path { stroke: var(--panel); stroke-width: 2; cursor: default; }
.pie svg path:hover { opacity: .85; }
.pie .n { font-size: 12px; color: var(--ink-3); margin-top: 2px; }
.pie .n b { color: var(--ink-2); font-weight: 600; }
.pie text.center { fill: var(--ink-2); font-size: 13px; font-weight: 600; }
.leg { width: 100%; border-collapse: collapse; font-size: 12px; }
.leg td { padding: 2px 4px; border: 0; text-align: right; white-space: nowrap; color: var(--ink-2); }
.leg td.name { text-align: left; white-space: normal; color: var(--ink); font-family: "IBM Plex Sans", -apple-system, sans-serif; }
.leg th { padding: 0 4px 2px; border: 0; font-weight: 500; color: var(--ink-3); font-size: 11px; text-align: right; }
.leg .sw { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; vertical-align: 0; }
.tip { position: fixed; pointer-events: none; background: var(--panel); border: 1px solid var(--line); color: var(--ink);
  border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); display: none; z-index: 9; }
.toggle { font-weight: 400; font-size: 13px; color: var(--ink-3); margin-left: 10px; cursor: pointer; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .cols { grid-template-columns: 1fr; } }
.col h4 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-2); }
.ex { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; min-width: 0; }
.ex-head { display: flex; flex-wrap: wrap; gap: 10px; align-items: baseline; font-size: 13px; color: var(--ink-3); }
.ex-head b { color: var(--ink); font-weight: 600; }
.player { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
.player span { font-size: 12px; color: var(--ink-3); width: 66px; flex: none; }
.player audio { width: 100%; height: 34px; min-width: 0; }
.qa { display: grid; grid-template-columns: 104px 1fr; gap: 6px 10px; margin-top: 10px; font-size: 14px; }
.k { color: var(--ink-3); font-size: 12px; padding-top: 2px; }
.v { min-width: 0; overflow-wrap: anywhere; }
.q { display: none; color: var(--ink-3); font-size: 12px; font-style: italic; }
body.show-q .q { display: block; }
.pred { margin-top: 3px; background: var(--pred); border-radius: 4px; padding: 1px 6px; font-size: 13px; }
</style>
</head>
<body>
<main>
<button class="theme" onclick="toggleTheme()">Light / dark</button>
<h1>KidSpeak datasets</h1>
<p class="sub">Three child-speech corpora turned into multi-turn audio question answering, split 7:3 into train / test <b>by child</b>.</p>

<h2>At a glance</h2>
<div class="table-wrap"><table>
<thead><tr><th>Dataset</th><th class="r">Children<br>train / test</th><th class="r">Clips<br>train / test</th><th class="r">Hours</th><th>Speech</th><th>Tasks</th></tr></thead>
<tbody>%ROWS%</tbody></table></div>
<ul class="notes">
  <li><b>One sample</b> = one audio clip + a multi-turn dialogue. Questions come from templates; answers come from the metadata.</li>
  <li><b>Labels describe the child</b> (age, gender, disorder, native language), so all clips of a child share them. Only the transcription changes per clip.</li>
  <li><b>The disorder question is balanced</b> within each split (about as many typical as disordered clips). All clips are still used for the other tasks.</li>
</ul>

<div role="tablist">%TABS%</div>
%PANELS%
</main>
<div class="tip" id="tip"></div>
<script>
const STATS = %STATS%;
const tip = document.getElementById('tip');
function toggleTheme() {
  const r = document.documentElement;
  const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = dark ? 'light' : 'dark';
}
function show(ds) {
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('on', p.id === ds));
  document.querySelectorAll('[role=tab]').forEach(t => t.setAttribute('aria-selected', t.dataset.ds === ds));
  history.replaceState(null, '', '#' + ds);
}
function chart(fig) {
  const ds = fig.dataset.ds, key = fig.dataset.key, cats = JSON.parse(fig.dataset.cats), S = ['train', 'test'];
  const counts = S.map(s => cats.map(c => (STATS[ds][s].dist[key] || {})[c] || 0));
  const totals = counts.map(r => r.reduce((a, b) => a + b, 0));
  const color = i => `var(--c${i % 8 + 1})`;
  const ns = 'http://www.w3.org/2000/svg';
  const pies = document.createElement('div'); pies.className = 'pies';
  S.forEach((s, j) => {
    const box = document.createElement('div'); box.className = 'pie';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 120 120'); svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${fig.querySelector('figcaption').textContent}, ${s}`);
    const R = 56, r = 32, cx = 60, cy = 60; let a0 = -Math.PI / 2;
    counts[j].forEach((v, i) => {
      if (!v) return;
      const frac = v / totals[j], a1 = a0 + frac * 2 * Math.PI;
      const p = document.createElementNS(ns, 'path');
      if (frac >= 0.9999) {
        p.setAttribute('d', `M${cx - R},${cy} a${R},${R} 0 1,0 ${2 * R},0 a${R},${R} 0 1,0 ${-2 * R},0 ` +
          `M${cx - r},${cy} a${r},${r} 0 1,1 ${2 * r},0 a${r},${r} 0 1,1 ${-2 * r},0`);
        p.setAttribute('fill-rule', 'evenodd');
      } else {
        const large = a1 - a0 > Math.PI ? 1 : 0, pt = (rad, a) => [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
        const [x0, y0] = pt(R, a0), [x1, y1] = pt(R, a1), [x2, y2] = pt(r, a1), [x3, y3] = pt(r, a0);
        p.setAttribute('d', `M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${large} 0 ${x3},${y3} Z`);
      }
      p.setAttribute('fill', color(i));
      p.addEventListener('mousemove', e => {
        tip.style.display = 'block';
        tip.innerHTML = `<b>${cats[i]}</b> · ${s}<br>${(frac * 100).toFixed(1)}% (${v.toLocaleString()} of ${totals[j].toLocaleString()})`;
        tip.style.left = Math.min(e.clientX + 12, innerWidth - 220) + 'px'; tip.style.top = e.clientY + 12 + 'px';
      });
      p.addEventListener('mouseleave', () => tip.style.display = 'none');
      svg.appendChild(p); a0 = a1;
    });
    const t = document.createElementNS(ns, 'text');
    t.setAttribute('x', cx); t.setAttribute('y', cy); t.setAttribute('text-anchor', 'middle');
    t.setAttribute('dominant-baseline', 'central'); t.setAttribute('class', 'center'); t.textContent = s;
    svg.appendChild(t);
    box.appendChild(svg);
    const n = document.createElement('div'); n.className = 'n';
    n.innerHTML = `<b>${totals[j].toLocaleString()}</b> ${key === 'duration' || key === 'subset' ? 'clips' : 'answers'}`;
    box.appendChild(n); pies.appendChild(box);
  });
  fig.appendChild(pies);
  const pct = (j, i) => totals[j] ? (counts[j][i] / totals[j] * 100).toFixed(counts[j][i] / totals[j] < .1 ? 1 : 0) + '%' : '-';
  const tbl = document.createElement('table'); tbl.className = 'leg';
  tbl.innerHTML = '<thead><tr><th></th><th>train</th><th>test</th></tr></thead><tbody>' + cats.map((c, i) =>
    `<tr><td class="name"><span class="sw" style="background:${color(i)}"></span>${c}</td><td>${pct(0, i)}</td><td>${pct(1, i)}</td></tr>`).join('') + '</tbody>';
  fig.appendChild(tbl);
}
document.querySelectorAll('figure.chart').forEach(chart);
show(location.hash.slice(1) in STATS ? location.hash.slice(1) : Object.keys(STATS)[0]);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', default='dataset/json')
    parser.add_argument('--audio_root', default='dataset')
    parser.add_argument('--pred_dir', default=None, help='inference outputs on test, to show the model answers')
    parser.add_argument('--out_dir', default='.', help='dataset_overview.html and demos/ are written here')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--artifact', action='store_true', help='write a Claude Artifact page (no document skeleton)')
    args = parser.parse_args()

    prompts = load_prompts(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts'))
    data, stats = collect(args.json_dir, args.audio_root, prompts)
    preds = load_predictions(args.pred_dir)
    shutil.rmtree(os.path.join(args.out_dir, 'demos'), ignore_errors=True)
    examples = {ds: {s: [] for s in SPLITS} for ds in DATASETS}
    for ds in DATASETS:
        for i, (split, sample) in enumerate(pick_examples(ds, data, args.seed, prompts)):
            examples[ds][split].append(example_card(ds, split, i, sample, args.audio_root, args.out_dir, preds, prompts))
    out = os.path.join(args.out_dir, 'dataset_overview.html')
    render(stats, examples, out, args.artifact)
    print(f'[!] Wrote {out} and demos/')


if __name__ == '__main__':
    main()
