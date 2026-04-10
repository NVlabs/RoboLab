"""
DROID Dataset Distribution Analysis

Compares robolab benchmark tasks against the DROID dataset to quantify overlap
in object vocabulary, per-task coverage, and task complexity distribution.

Analyses:
    1. Object Vocabulary Overlap (Jaccard / Set Intersection)
    2. Per-Task "In-Distribution" Score
    3. Complexity Distribution Comparison (ratio)

Usage:
    python analysis/droid_dataset/droid_dataset_distribution.py
    python analysis/droid_dataset/droid_dataset_distribution.py --metadata path/to/task_metadata.json
    python analysis/droid_dataset/droid_dataset_distribution.py --save-dir analysis/droid_dataset/output
    python analysis/droid_dataset/droid_dataset_distribution.py --no-plot
"""

import json
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn2, venn2_circles
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional


########################################################
# DROID Dataset Constants (from DROID dataset analysis)
# Total: 31,308 tasks
########################################################

DROID_TOTAL_TASKS = 31_308

DROID_OBJECTS = {
    'unknown': 4789, 'bowl': 3819, 'cup': 3621, 'box': 2271,
    'orange': 2007, 'lid': 1936, 'bottle': 1799, 'pot': 1782,
    'towel': 1542, 'drawer': 1513, 'plate': 1502, 'marker': 1499,
    'mug': 1380, 'cabinet': 1145, 'spoon': 1117, 'toy': 1024,
    'cloth': 958, 'block': 922, 'pen': 806, 'bag': 718,
    'plush': 682, 'container': 667, 'door': 644, 'shelf': 609,
    'paper': 603, 'tray': 583, 'switch': 505, 'glass': 489,
    'bin': 482, 'basket': 477,
}

DROID_MOVEMENT_TYPES = {
    'place/put': 15500, 'pick/grasp': 7500, 'move/reposition': 3500,
    'open': 2000, 'close': 1500, 'other': 1000, 'push/press': 800,
    'turn/rotate': 700, 'wipe/clean': 400, 'pour': 300,
}

DROID_TARGET_LOCATIONS = {
    'receptacle': 10000, 'table/desk/counter': 8500,
    'directional': 7000, 'unspecified': 5000, 'container': 3500,
    'vertical': 2500, 'appliance': 1500, 'drawer': 1000,
    'cabinet/cupboard': 800, 'sink/basin': 500,
}

DROID_COMPLEXITY = {
    'single-step': 18200,
    'two-step': 10890,
    'multi-step': 2217,
}


########################################################
# Styling
########################################################

COLORS = {
    'droid': '#4A90D9',
    'benchmark': '#5CB85C',
    'overlap': '#6A5ACD',
    'droid_only': '#B0BEC5',
    'benchmark_only': '#FFB74D',
    'simple': '#66BB6A',
    'moderate': '#FFA726',
    'complex': '#EF5350',
}


def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


########################################################
# Object Mapping
########################################################

DROID_OBJECT_SET = set(DROID_OBJECTS.keys()) - {'unknown'}

BENCHMARK_SYNONYM_MAP = {
    # Bowl
    'serving_bowl': 'bowl', 'wooden_bowl': 'bowl',
    'blackandbrassbowl_large': 'bowl',
    # Mug
    'ceramic_mug': 'mug', 'red_mug': 'mug',
    'upright_white_mug': 'mug', 'sideways_white_mug': 'mug',
    # Bin
    'grey_bin': 'bin', 'right_bin': 'bin', 'left_bin': 'bin', 'center_bin': 'bin',
    'grey_bin_left': 'bin', 'grey_bin_right': 'bin',
    # Plate
    'plate_large': 'plate', 'plate_small': 'plate', 'clay_plates': 'plate',
    # Spoon
    'spoon_big': 'spoon', 'pink_spaghetti_spoon': 'spoon',
    'green_serving_spoon': 'spoon', 'red_serving_spoon': 'spoon',
    'wooden_spoons': 'spoon',
    # Cup
    'measuring_cup': 'cup', 'measuring_cups': 'cup', 'yogurt_cup': 'cup',
    # Bottle
    'bbq_sauce_bottle': 'bottle', 'ketchup_bottle': 'bottle',
    'mustard_bottle': 'bottle', 'mayonnaise_bottle': 'bottle',
    'mustard': 'bottle', 'ranch_dressing': 'bottle', 'soft_scrub': 'bottle',
    'whitepackerbottle': 'bottle', 'utilityjug': 'bottle', 'milkjug': 'bottle',
    # Block
    'yellow_block': 'block', 'red_block': 'block',
    'green_block': 'block', 'blue_block': 'block',
    'wood_block': 'block', 'frozen_vegetable_block': 'block',
    # Box
    'storage_box': 'box', 'cheez_it': 'box', 'cubebox': 'box',
    'sugar_box': 'box', 'raisin_box': 'box', 'chocolate_pudding': 'box',
    # Pot
    'anza_medium': 'pot', 'coffee_pot': 'pot',
    # Toy
    'rubiks_cube': 'toy', 'rubiks_cube_middle': 'toy',
    'rubiks_cube_top': 'toy', 'rubiks_cube_bottom': 'toy',
    'lizard_figurine': 'toy', 'birdhouse': 'toy',
    # Shelf
    'large_storage_rack': 'shelf', 'rack_l04': 'shelf',
    # Marker
    'dry_erase_marker': 'marker',
    # Container
    'purple_crate': 'container', 'pitcher': 'container',
    'tomato_soup_can': 'container', 'tuna_can': 'container',
    'spam_can': 'container', 'alphabet_soup_can': 'container',
    'canned_tuna': 'container', 'tomato_sauce_can': 'container',
    'coffee_can': 'container', 'milk_carton': 'container',
    'orange_juice_carton': 'container',
    'plasticpail': 'container', 'plasticjerrican': 'container',
    'squarepail': 'container', 'screwtoppail': 'container',
    # Glass
    'glasses': 'glass',
    # Pen
    'crabbypenholder': 'pen',
    # Shelf (wire)
    'wireshelving': 'shelf',
    # Orange variant without underscore
    'orange2': 'orange',
}


def normalize_object_name(name: str) -> str:
    """Strip trailing numeric suffixes like _01, _02."""
    return re.sub(r'_\d+$', '', name.strip().lower())


def map_to_droid_category(obj_name: str) -> Optional[str]:
    """Map a benchmark object name to a DROID object category, or None."""
    normalized = normalize_object_name(obj_name)
    if normalized in DROID_OBJECT_SET:
        return normalized
    if normalized in BENCHMARK_SYNONYM_MAP:
        return BENCHMARK_SYNONYM_MAP[normalized]
    # Try stripping _<letter><digits> suffix (e.g. _a01, _a02)
    base = re.sub(r'_[a-z]\d+$', '', normalized)
    if base != normalized:
        if base in DROID_OBJECT_SET:
            return base
        if base in BENCHMARK_SYNONYM_MAP:
            return BENCHMARK_SYNONYM_MAP[base]
    return None


########################################################
# Instruction Parsing
########################################################

def extract_action_type(instruction: str) -> str:
    """Map instruction text to a DROID movement type."""
    instr = instruction.lower().strip()
    if any(w in instr for w in ['reorient', 'upright']):
        return 'turn/rotate'
    if any(w in instr for w in ['pick up', 'grab ', 'select ']):
        return 'pick/grasp'
    if re.match(r'^take\b', instr) or 'take out' in instr or 'take off' in instr:
        return 'pick/grasp'
    if 'unstack' in instr:
        return 'pick/grasp'
    if re.match(r'^move\b', instr) or 'reposition' in instr:
        return 'move/reposition'
    return 'place/put'


def extract_target_location(instruction: str) -> str:
    """Map instruction text to a DROID target location category."""
    instr = instruction.lower()
    if any(w in instr for w in ['on the shelf', 'on shelf', 'off the shelf', 'off shelf']):
        return 'vertical'
    if any(w in instr for w in ['on the table', 'on table']):
        return 'table/desk/counter'
    if any(w in instr for w in ['left of', 'right of', 'behind', 'in front of',
                                 'above', 'center of', 'in the left', 'in the right']):
        return 'directional'
    if any(w in instr for w in ['in the bowl', 'in the bin', 'in the pot',
                                 'in the mug', 'in the crate', 'in the pail',
                                 'in the container', 'into the', 'in the grey',
                                 'in the square']):
        return 'container'
    if any(w in instr for w in ['on the plate', 'on plate', 'on the box',
                                 'on the raisin']):
        return 'receptacle'
    if any(w in instr for w in ['pick', 'grab', 'select']):
        return 'unspecified'
    return 'receptacle'


########################################################
# Analysis 1: Object Vocabulary Overlap
########################################################

def compute_object_overlap(tasks_data: List[Dict]) -> Dict[str, Any]:
    benchmark_objects_raw = set()
    droid_category_to_task_indices: Dict[str, set] = defaultdict(set)

    for i, task in enumerate(tasks_data):
        objects_str = task.get('contact_objects', '')
        if not objects_str:
            continue
        for obj in objects_str.split(','):
            obj = obj.strip()
            if obj.lower() == 'table':
                continue
            normalized = normalize_object_name(obj)
            benchmark_objects_raw.add(normalized)
            droid_cat = map_to_droid_category(obj)
            if droid_cat:
                droid_category_to_task_indices[droid_cat].add(i)

    benchmark_droid_categories = set(droid_category_to_task_indices.keys())
    unmapped = {obj for obj in benchmark_objects_raw if map_to_droid_category(obj) is None}

    overlap = DROID_OBJECT_SET & benchmark_droid_categories
    droid_only = DROID_OBJECT_SET - benchmark_droid_categories

    all_items = DROID_OBJECT_SET | benchmark_droid_categories | unmapped
    jaccard = len(overlap) / len(all_items) if all_items else 0
    coverage = len(overlap) / len(DROID_OBJECT_SET) if DROID_OBJECT_SET else 0

    benchmark_task_counts = {
        cat: len(indices) for cat, indices in droid_category_to_task_indices.items()
    }

    return {
        'overlap': sorted(overlap),
        'droid_only': sorted(droid_only),
        'benchmark_only': sorted(unmapped),
        'jaccard': jaccard,
        'droid_coverage': coverage,
        'num_benchmark_raw': len(benchmark_objects_raw),
        'num_benchmark_mapped': len(benchmark_droid_categories),
        'num_droid': len(DROID_OBJECT_SET),
        'benchmark_task_counts': benchmark_task_counts,
    }


def print_object_overlap(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("ANALYSIS 1: OBJECT VOCABULARY OVERLAP")
    print("=" * 80)
    print(f"Benchmark unique objects (normalized): {results['num_benchmark_raw']}")
    print(f"DROID object categories (excl. 'unknown'): {results['num_droid']}")
    print(f"Overlapping categories: {len(results['overlap'])}")
    print(f"DROID-only categories: {len(results['droid_only'])}")
    print(f"Benchmark-only objects: {len(results['benchmark_only'])}")
    print(f"Jaccard similarity: {results['jaccard']:.3f}")
    print(f"DROID coverage: {results['droid_coverage']:.1%} "
          f"({len(results['overlap'])}/{results['num_droid']})")

    print(f"\nOverlapping: {', '.join(results['overlap'])}")
    print(f"DROID-only:  {', '.join(results['droid_only'])}")
    print(f"Benchmark-only: {', '.join(results['benchmark_only'])}")

    print(f"\nBenchmark tasks per DROID category:")
    sorted_cats = sorted(results['benchmark_task_counts'].keys(),
                         key=lambda x: -results['benchmark_task_counts'][x])
    for cat in sorted_cats:
        count = results['benchmark_task_counts'][cat]
        droid_count = DROID_OBJECTS.get(cat, 0)
        print(f"  {cat:15s}: {count:3d} benchmark tasks  |  "
              f"{droid_count:5d} DROID tasks")


def plot_object_overlap(results: Dict[str, Any], save_path: Optional[str] = None):
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 2]}
    )

    # Left panel: DROID object categories colored by overlap status
    droid_cats = sorted(DROID_OBJECT_SET,
                        key=lambda x: DROID_OBJECTS.get(x, 0))
    droid_freqs = [DROID_OBJECTS.get(c, 0) for c in droid_cats]
    colors = [
        COLORS['overlap'] if c in results['overlap'] else COLORS['droid_only']
        for c in droid_cats
    ]

    ax1.barh(range(len(droid_cats)), droid_freqs, color=colors,
             edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(droid_cats)))
    ax1.set_yticklabels(droid_cats, fontsize=10)
    ax1.set_xlabel('DROID Task Count')
    ax1.set_title('DROID Object Categories\n(colored by benchmark overlap)',
                   fontweight='bold', fontsize=13)

    for i, (cat, freq) in enumerate(zip(droid_cats, droid_freqs)):
        bench_count = results['benchmark_task_counts'].get(cat, 0)
        if bench_count > 0:
            ax1.text(freq + 50, i, f'{bench_count} bench tasks',
                     va='center', fontsize=8, color=COLORS['benchmark'],
                     fontweight='bold')

    overlap_patch = mpatches.Patch(
        color=COLORS['overlap'],
        label=f"In both ({len(results['overlap'])})")
    droid_patch = mpatches.Patch(
        color=COLORS['droid_only'],
        label=f"DROID only ({len(results['droid_only'])})")
    ax1.legend(handles=[overlap_patch, droid_patch],
               loc='lower right', fontsize=10)

    # Right panel: summary stats and object lists
    ax2.axis('off')
    lines = [
        "Object Vocabulary Overlap",
        "─" * 35, "",
        f"Benchmark objects:  {results['num_benchmark_raw']}",
        f"DROID categories:   {results['num_droid']}",
        f"Overlapping:        {len(results['overlap'])}", "",
        f"Jaccard similarity: {results['jaccard']:.3f}",
        f"DROID coverage:     {results['droid_coverage']:.1%}", "",
        "─" * 35,
        f"DROID-only ({len(results['droid_only'])}):",
    ]
    for obj in results['droid_only']:
        lines.append(f"  • {obj}")
    lines.append("")
    lines.append(f"Benchmark-only ({len(results['benchmark_only'])}):")
    bonly = sorted(results['benchmark_only'])
    for obj in bonly[:15]:
        lines.append(f"  • {obj}")
    if len(bonly) > 15:
        lines.append(f"  ... and {len(bonly) - 15} more")

    ax2.text(0.05, 0.95, "\n".join(lines), transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                       edgecolor='#DDDDDD'))

    plt.suptitle('Benchmark vs DROID: Object Vocabulary Overlap',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_object_venn(results: Dict[str, Any], save_path: Optional[str] = None):
    setup_plot_style()

    droid_only = results['droid_only']
    overlap = results['overlap']
    bench_only = sorted(results['benchmark_only'])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.25)

    # --- Top: Venn diagram ---
    ax_venn = fig.add_subplot(gs[0])

    v = venn2(
        subsets=(len(droid_only), len(bench_only), len(overlap)),
        set_labels=('', ''),
        ax=ax_venn,
    )

    patch_styles = {
        '10': (COLORS['droid'], 0.40),
        '01': (COLORS['benchmark'], 0.40),
        '11': (COLORS['overlap'], 0.55),
    }
    for pid, (color, alpha) in patch_styles.items():
        p = v.get_patch_by_id(pid)
        if p:
            p.set_color(color)
            p.set_alpha(alpha)
            p.set_edgecolor('white')

    c = venn2_circles(
        subsets=(len(droid_only), len(bench_only), len(overlap)),
        ax=ax_venn, linewidth=2, color='#444444',
    )

    # Count labels inside circles
    for pid, count in [('10', len(droid_only)),
                       ('01', len(bench_only)),
                       ('11', len(overlap))]:
        lbl = v.get_label_by_id(pid)
        if lbl:
            lbl.set_text(str(count))
            lbl.set_fontsize(20)
            lbl.set_fontweight('bold')
            lbl.set_color('#222222')

    # Set labels
    ax_venn.text(0.22, 0.88,
                 f'DROID\n({results["num_droid"]} categories)',
                 ha='center', fontsize=14, fontweight='bold',
                 color=COLORS['droid'], transform=ax_venn.transAxes)
    ax_venn.text(0.78, 0.88,
                 f'Benchmark\n({results["num_benchmark_raw"]} objects)',
                 ha='center', fontsize=14, fontweight='bold',
                 color=COLORS['benchmark'], transform=ax_venn.transAxes)

    stats = (f"Jaccard similarity: {results['jaccard']:.3f}        "
             f"DROID coverage: {results['droid_coverage']:.1%}")
    ax_venn.text(0.5, 0.03, stats, ha='center', fontsize=11,
                 style='italic', color='#555555',
                 transform=ax_venn.transAxes)

    ax_venn.set_title('Object Vocabulary Overlap: DROID vs Benchmark',
                      fontsize=16, fontweight='bold', pad=15)

    # --- Bottom: three-column object lists ---
    ax_list = fig.add_subplot(gs[1])
    ax_list.axis('off')

    col_x = [0.04, 0.38, 0.7]
    headers = [
        (f'DROID only ({len(droid_only)})', COLORS['droid']),
        (f'Both ({len(overlap)})', COLORS['overlap']),
        (f'Benchmark only ({len(bench_only)})', COLORS['benchmark']),
    ]
    lists = [droid_only, overlap, bench_only]

    for col_idx, ((header, color), items) in enumerate(zip(headers, lists)):
        x = col_x[col_idx]
        ax_list.text(x, 0.95, header, transform=ax_list.transAxes,
                     fontsize=12, fontweight='bold', color=color, va='top')
        ax_list.plot([x, x + 0.24], [0.90, 0.90],
                     transform=ax_list.transAxes, color=color,
                     linewidth=1.5, clip_on=False)

        max_show = 20
        for i, item in enumerate(items[:max_show]):
            y = 0.85 - i * 0.042
            ax_list.text(x + 0.01, y, f'• {item}', transform=ax_list.transAxes,
                         fontsize=9.5, va='top', fontfamily='monospace')
        if len(items) > max_show:
            y = 0.85 - max_show * 0.042
            ax_list.text(x + 0.01, y,
                         f'  ... +{len(items) - max_show} more',
                         transform=ax_list.transAxes,
                         fontsize=9, va='top', color='#888888',
                         style='italic')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
# Analysis 2: Per-Task In-Distribution Score
########################################################

def compute_per_task_scores(tasks_data: List[Dict]) -> List[Dict]:
    max_obj_freq = max(DROID_OBJECTS.values())
    max_action_freq = max(DROID_MOVEMENT_TYPES.values())
    max_location_freq = max(DROID_TARGET_LOCATIONS.values())

    results = []
    for task in tasks_data:
        objects_str = task.get('contact_objects', '')
        obj_scores = []
        matched_objects = 0
        total_objects = 0

        if objects_str:
            for obj in objects_str.split(','):
                obj = obj.strip()
                if obj.lower() == 'table':
                    continue
                total_objects += 1
                droid_cat = map_to_droid_category(obj)
                if droid_cat and droid_cat in DROID_OBJECTS:
                    obj_scores.append(DROID_OBJECTS[droid_cat] / max_obj_freq)
                    matched_objects += 1
                else:
                    obj_scores.append(0.0)

        avg_obj_score = float(np.mean(obj_scores)) if obj_scores else 0.0
        obj_match_ratio = matched_objects / total_objects if total_objects > 0 else 0.0

        instruction = task.get('instruction', '')
        action_type = extract_action_type(instruction)
        action_score = DROID_MOVEMENT_TYPES.get(action_type, 0) / max_action_freq

        target_loc = extract_target_location(instruction)
        location_score = DROID_TARGET_LOCATIONS.get(target_loc, 0) / max_location_freq

        # Weighted combination: objects most important, then action, then location
        total_score = (0.5 * avg_obj_score
                       + 0.3 * action_score
                       + 0.2 * location_score)

        results.append({
            'task_name': task.get('task_name', 'Unknown'),
            'instruction': instruction,
            'object_score': avg_obj_score,
            'object_match_ratio': obj_match_ratio,
            'action_type': action_type,
            'action_score': action_score,
            'target_location': target_loc,
            'location_score': location_score,
            'total_score': total_score,
            'difficulty_label': task.get('difficulty_label', 'simple'),
            'difficulty_score': task.get('difficulty_score', 0),
            'num_subtasks': task.get('num_subtasks', 0),
        })

    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results


def print_per_task_scores(results: List[Dict]):
    print("\n" + "=" * 80)
    print("ANALYSIS 2: PER-TASK IN-DISTRIBUTION SCORE")
    print("=" * 80)
    print("score = 0.5 * object_freq + 0.3 * action_freq + 0.2 * location_freq")
    print("Each component normalized to [0,1] relative to DROID max frequency.\n")

    scores = [r['total_score'] for r in results]
    print(f"Average score: {np.mean(scores):.3f}")
    print(f"Median score:  {np.median(scores):.3f}")
    print(f"Std dev:       {np.std(scores):.3f}")
    print(f"Range:         [{min(scores):.3f}, {max(scores):.3f}]")

    print("\nBy difficulty level:")
    for label in ['simple', 'moderate', 'complex']:
        subset = [r['total_score'] for r in results if r['difficulty_label'] == label]
        if subset:
            print(f"  {label:10s}: mean={np.mean(subset):.3f}  "
                  f"median={np.median(subset):.3f}  n={len(subset)}")

    print(f"\nTop 10 most in-distribution tasks:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:2d}. {r['task_name']:<42s} score={r['total_score']:.3f}  "
              f"obj={r['object_score']:.2f} act={r['action_score']:.2f} "
              f"loc={r['location_score']:.2f}  [{r['difficulty_label']}]")

    print(f"\nBottom 10 least in-distribution tasks:")
    for i, r in enumerate(results[-10:], 1):
        print(f"  {i:2d}. {r['task_name']:<42s} score={r['total_score']:.3f}  "
              f"obj={r['object_score']:.2f} act={r['action_score']:.2f} "
              f"loc={r['location_score']:.2f}  [{r['difficulty_label']}]")


def plot_per_task_scores(results: List[Dict], save_path: Optional[str] = None):
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]}
    )

    difficulty_order = ['simple', 'moderate', 'complex']
    difficulty_colors = [COLORS[d] for d in difficulty_order]

    # Left panel: box plot + strip by difficulty level
    data_by_diff = []
    for label in difficulty_order:
        subset = [r['total_score'] for r in results if r['difficulty_label'] == label]
        data_by_diff.append(subset)

    bp = ax1.boxplot(data_by_diff, vert=True, patch_artist=True,
                     labels=[f"{d}\n(n={len(s)})" for d, s in
                             zip(difficulty_order, data_by_diff)],
                     widths=0.55, showfliers=False)
    for patch, color in zip(bp['boxes'], difficulty_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    for i, (label, subset) in enumerate(zip(difficulty_order, data_by_diff)):
        x = np.random.default_rng(42).normal(i + 1, 0.07, size=len(subset))
        ax1.scatter(x, subset, alpha=0.6, s=30, color=difficulty_colors[i],
                    edgecolor='white', linewidth=0.5, zorder=5)

    ax1.set_ylabel('In-Distribution Score', fontsize=12)
    ax1.set_title('Score Distribution by Difficulty', fontweight='bold')
    ax1.set_ylim(-0.02, 0.85)

    # Right panel: top 10 and bottom 10 tasks
    n_show = 10
    top_tasks = results[:n_show]
    bottom_tasks = results[-n_show:]

    labels_list = []
    scores_list = []
    bar_colors = []

    for r in reversed(bottom_tasks):
        short = r['task_name'].replace('Task', '')
        labels_list.append(short)
        scores_list.append(r['total_score'])
        bar_colors.append(COLORS[r['difficulty_label']])

    sep_idx = len(labels_list)
    labels_list.append('')
    scores_list.append(0)
    bar_colors.append('white')

    for r in reversed(top_tasks):
        short = r['task_name'].replace('Task', '')
        labels_list.append(short)
        scores_list.append(r['total_score'])
        bar_colors.append(COLORS[r['difficulty_label']])

    y_pos = range(len(labels_list))
    ax2.barh(y_pos, scores_list, color=bar_colors,
             edgecolor='white', linewidth=0.5, height=0.75)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels_list, fontsize=9)
    ax2.set_xlabel('In-Distribution Score')
    ax2.set_title('Highest & Lowest DROID Coverage', fontweight='bold')

    for i, (score, label) in enumerate(zip(scores_list, labels_list)):
        if label:
            ax2.text(score + 0.008, i, f'{score:.2f}',
                     va='center', fontsize=8)

    ax2.axhline(y=sep_idx, color='gray', linestyle='--', alpha=0.5)
    ax2.text(max(scores_list) * 0.5, sep_idx,
             '— gap —', ha='center', va='center',
             fontsize=8, color='gray', style='italic',
             bbox=dict(facecolor='white', edgecolor='none', pad=1))

    patches = [mpatches.Patch(color=COLORS[d], label=d) for d in difficulty_order]
    ax2.legend(handles=patches, loc='center right', fontsize=9)

    plt.suptitle('Per-Task In-Distribution Score (Benchmark vs DROID)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
# Analysis 3: Complexity Distribution Comparison
########################################################

def compute_complexity_comparison(tasks_data: List[Dict]) -> Dict[str, Any]:
    benchmark_counts: Counter = Counter()
    for task in tasks_data:
        n = task.get('num_subtasks', 0)
        if n <= 1:
            benchmark_counts['single-step'] += 1
        elif n == 2:
            benchmark_counts['two-step'] += 1
        else:
            benchmark_counts['multi-step'] += 1

    total_bench = sum(benchmark_counts.values())
    total_droid = sum(DROID_COMPLEXITY.values())
    categories = ['single-step', 'two-step', 'multi-step']

    bench_ratios = {c: benchmark_counts[c] / total_bench for c in categories}
    droid_ratios = {c: DROID_COMPLEXITY[c] / total_droid for c in categories}

    return {
        'categories': categories,
        'benchmark_counts': dict(benchmark_counts),
        'droid_counts': dict(DROID_COMPLEXITY),
        'benchmark_ratios': bench_ratios,
        'droid_ratios': droid_ratios,
        'total_benchmark': total_bench,
        'total_droid': total_droid,
    }


def print_complexity_comparison(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("ANALYSIS 3: COMPLEXITY DISTRIBUTION COMPARISON (RATIO)")
    print("=" * 80)
    print(f"Benchmark total: {results['total_benchmark']} tasks")
    print(f"DROID total:     {results['total_droid']:,} tasks\n")

    header = (f"{'Category':<15s} {'Benchmark':>15s} {'DROID':>15s} "
              f"{'Δ (bench−droid)':>18s}")
    print(header)
    print("-" * 66)
    for cat in results['categories']:
        b_ratio = results['benchmark_ratios'][cat]
        d_ratio = results['droid_ratios'][cat]
        b_count = results['benchmark_counts'][cat]
        d_count = results['droid_counts'][cat]
        delta = b_ratio - d_ratio
        sign = '+' if delta >= 0 else ''
        print(f"{cat:<15s} {b_ratio:>8.1%} ({b_count:>3d})  "
              f"{d_ratio:>8.1%} ({d_count:>5d})   {sign}{delta:>8.1%}")


def plot_complexity_comparison(results: Dict[str, Any],
                               save_path: Optional[str] = None):
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = results['categories']
    bench_pcts = [results['benchmark_ratios'][c] * 100 for c in categories]
    droid_pcts = [results['droid_ratios'][c] * 100 for c in categories]

    x = np.arange(len(categories))
    width = 0.32

    bars_droid = ax.bar(
        x - width / 2, droid_pcts, width, color=COLORS['droid'],
        label=f'DROID (n={results["total_droid"]:,})',
        edgecolor='white', linewidth=0.5)
    bars_bench = ax.bar(
        x + width / 2, bench_pcts, width, color=COLORS['benchmark'],
        label=f'Benchmark (n={results["total_benchmark"]})',
        edgecolor='white', linewidth=0.5)

    for bars in [bars_droid, bars_bench]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.8,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    display_labels = [
        'Single-step\n(≤1 subtask)',
        'Two-step\n(2 subtasks)',
        'Multi-step\n(3+ subtasks)',
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.set_ylabel('Percentage of Tasks (%)', fontsize=12)
    ax.set_title('Task Complexity Distribution: Benchmark vs DROID',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, max(max(bench_pcts), max(droid_pcts)) + 12)
    ax.grid(axis='x', visible=False)

    # Annotate the deltas
    for i, cat in enumerate(categories):
        delta = bench_pcts[i] - droid_pcts[i]
        sign = '+' if delta >= 0 else ''
        color = '#D9534F' if abs(delta) > 10 else '#888888'
        y_pos = max(bench_pcts[i], droid_pcts[i]) + 6
        ax.text(x[i], y_pos, f'Δ {sign}{delta:.1f}pp',
                ha='center', fontsize=9, color=color, style='italic')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
# DROID Raw Task Analysis (from tasks.jsonl)
########################################################

STOPWORDS = {
    'a', 'an', 'the', 'in', 'on', 'to', 'of', 'it', 'and', 'or', 'is',
    'that', 'this', 'its', 'with', 'for', 'from', 'at', 'by', 'are',
    'be', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'shall', 'can', 'there', 'their', 'them', 'they', 'he', 'she',
    'not', 'no', 'so', 'if', 'but', 'all', 'each', 'every', 'any',
    'both', 'either', 'neither', 'other', 'such', 'than', 'too',
    'very', 'just', 'also', 'then', 'up', 'out', 'off', 'down',
    'away', 'into', 'onto', 'over', 'under', 'about', 'after',
    'before', 'between', 'through', 'during', 'without', 'within',
    'along', 'around', 'across', 'more', 'back', 'next', 'one', 'two',
    'three', 'four', 'five', 'first', 'second', 'third',
}


def load_droid_tasks(filepath: str) -> List[str]:
    tasks = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_text = data.get('task', '').strip()
            if task_text:
                tasks.append(task_text)
    return tasks


def load_droid_language_annotations(filepath: str) -> List[List[str]]:
    """Load DROID language annotations JSON.

    Returns a list of episodes, where each episode is a list of 1-3
    instruction strings (one per annotator).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    episodes = []
    for _ep_id, annotations in data.items():
        instructions = []
        for key in ('language_instruction1', 'language_instruction2',
                    'language_instruction3'):
            text = annotations.get(key, '').strip()
            if text:
                instructions.append(text)
        if instructions:
            episodes.append(instructions)
    return episodes


def tokenize_instruction(text: str) -> List[str]:
    """Tokenize instruction into lowercase content words."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]


def compute_droid_word_frequencies(droid_tasks: List[str]) -> Counter:
    """Word frequency across DROID tasks (each word counted once per task)."""
    freq = Counter()
    for task in droid_tasks:
        freq.update(set(tokenize_instruction(task)))
    return freq


########################################################
# DROID Object Vocabulary Extraction
########################################################

NON_OBJECT_WORDS = {
    # --- Verbs / action words ---
    'put', 'place', 'pick', 'move', 'push', 'pull', 'slide', 'open', 'close',
    'turn', 'rotate', 'flip', 'press', 'grab', 'lift', 'take', 'remove',
    'stack', 'pour', 'wipe', 'clean', 'fold', 'unfold', 'sort', 'arrange',
    'hold', 'drop', 'throw', 'toss', 'insert', 'hang', 'switch', 'toggle',
    'adjust', 'bring', 'carry', 'set', 'get', 'make', 'keep', 'leave',
    'use', 'start', 'stop', 'begin', 'end', 'try', 'reach', 'touch',
    'squeeze', 'twist', 'shake', 'stir', 'spread', 'wrap', 'unwrap', 'peel',
    'cut', 'chop', 'screw', 'unscrew', 'lock', 'unlock', 'load', 'unload',
    'fill', 'empty', 'dump', 'sweep', 'scrub', 'rinse', 'dry', 'store',
    'retrieve', 'fetch', 'deliver', 'hand', 'pass', 'swap', 'exchange',
    'replace', 'return', 'clear', 'cover', 'uncover', 'plug', 'unplug',
    'connect', 'disconnect', 'attach', 'detach', 'assemble', 'disassemble',
    'position', 'reposition', 'straighten', 'align', 'tilt', 'lean', 'lay',
    'rest', 'stand', 'sit', 'raise', 'lower', 'extend', 'retract', 'roll',
    'spin', 'drag', 'nudge', 'tap', 'click', 'type', 'write', 'draw',
    'erase', 'mark', 'label', 'read', 'scan', 'check', 'inspect', 'examine',
    'look', 'see', 'find', 'locate', 'search', 'identify', 'select', 'choose',
    'ensure', 'confirm', 'verify', 'prepare', 'setup', 'organize', 'rearrange',
    'transfer', 'shift', 'relocate', 'flatten', 'smooth', 'pinch', 'grip',
    'release', 'clamp', 'secure', 'fasten', 'loosen', 'tighten', 'seal',
    'unseal', 'hang', 'separate', 'combine', 'mix', 'blend', 'wring',
    'collect', 'gather', 'pile', 'scatter', 'dispense', 'spray', 'pump',
    'reorient', 'orient', 'unstack', 'crumple', 'crunch',
    'put', 'placing', 'picking', 'moving', 'pushing', 'pulling', 'sliding',
    'opening', 'closing', 'turning', 'rotating', 'flipping', 'pressing',
    'grabbing', 'lifting', 'taking', 'removing', 'stacking', 'pouring',
    'wiping', 'cleaning', 'folding', 'unfolding', 'sorting', 'arranging',
    'holding', 'dropping', 'throwing', 'inserting', 'hanging', 'adjusting',
    'bringing', 'carrying', 'setting', 'keeping', 'leaving',
    'using', 'reaching', 'touching', 'squeezing', 'twisting', 'shaking',
    'stirring', 'spreading', 'wrapping', 'peeling', 'cutting',
    'placed', 'picked', 'moved', 'pushed', 'pulled', 'slid', 'opened',
    'closed', 'turned', 'rotated', 'flipped', 'pressed', 'grabbed',
    'lifted', 'taken', 'removed', 'stacked', 'poured', 'wiped', 'cleaned',
    'folded', 'sorted', 'arranged', 'held', 'dropped', 'thrown',
    'inserted', 'adjusted', 'brought', 'carried', 'kept',
    # --- Colors ---
    'red', 'blue', 'green', 'yellow', 'white', 'black', 'pink', 'purple',
    'brown', 'grey', 'gray', 'silver', 'gold', 'beige', 'tan', 'orange',
    'clear', 'transparent', 'dark', 'light', 'bright', 'colored',
    'multicolored', 'striped', 'patterned',
    # --- Materials / texture adjectives ---
    'wooden', 'plastic', 'metal', 'metallic', 'rubber', 'fabric', 'leather',
    'ceramic', 'cardboard', 'foam', 'silicone', 'stainless', 'steel',
    'aluminum', 'chrome', 'copper', 'iron', 'brass', 'velvet', 'cotton',
    'nylon', 'woven',
    # --- Size / shape / physical adjectives ---
    'soft', 'hard', 'flat', 'round', 'square', 'circular', 'rectangular',
    'triangular', 'long', 'short', 'tall', 'wide', 'narrow', 'thick', 'thin',
    'big', 'small', 'large', 'medium', 'tiny', 'huge', 'little', 'mini',
    'skinny', 'heavy', 'lightweight', 'bigger', 'smaller', 'larger', 'shorter',
    'taller', 'wider', 'longer', 'thinner', 'thicker',
    # --- State adjectives ---
    'new', 'old', 'dirty', 'wet', 'hot', 'cold', 'warm', 'cool',
    'full', 'broken', 'intact', 'correct', 'wrong', 'same', 'different',
    'similar', 'proper', 'appropriate', 'remaining', 'extra', 'additional',
    'available', 'upside', 'upright', 'sideways', 'crumpled',
    # --- Spatial / directional ---
    'left', 'right', 'top', 'bottom', 'front', 'rear', 'upper', 'lower',
    'above', 'below', 'beside', 'behind', 'inside', 'outside', 'center',
    'middle', 'edge', 'corner', 'side', 'near', 'far', 'forward', 'backward',
    'upward', 'downward', 'horizontal', 'vertical', 'diagonal', 'clockwise',
    'counterclockwise', 'inward', 'outward', 'facing', 'towards', 'toward',
    'closest', 'farthest', 'nearest', 'furthest', 'rightmost', 'leftmost',
    'topmost', 'bottommost', 'backwards', 'forwards', 'downwards', 'upwards',
    'closer', 'farther', 'overhead', 'underneath', 'atop', 'against',
    'opposite', 'surrounding', 'anticlockwise',
    # --- Quantity / order / misc function words ---
    'half', 'quarter', 'double', 'triple', 'single', 'pair', 'few', 'many',
    'several', 'multiple', 'enough', 'less', 'most', 'least', 'only',
    'exactly', 'approximately', 'roughly', 'slightly', 'completely',
    'fully', 'partially', 'slowly', 'quickly', 'gently', 'carefully',
    'way', 'thing', 'things', 'stuff', 'item', 'items', 'piece', 'pieces',
    'part', 'parts', 'another', 'new', 'same', 'like', 'some',
    'fourth', 'fifth', 'sixth', 'last', 'once', 'twice', 'again',
    'finally', 'respectively', 'together', 'straight',
    # --- Generic / abstract words ---
    'object', 'objects', 'contents', 'content', 'word', 'time', 'times',
    'unit', 'line', 'form', 'section', 'change', 'surface', 'shape',
    'shaped', 'circle', 'circles', 'degrees',
    # --- Pronouns / relative words ---
    'you', 'your', 'we', 'me', 'my', 'which', 'where', 'what', 'how',
    'who', 'whom', 'whose',
    # --- Additional adjectives ---
    'colourless', 'colorless', 'cylindrical', 'elastic', 'electric',
    'stuffed', 'sweet', 'maroon', 'liquid', 'denim', 'magenta',
    'turquoise', 'teal', 'ivory', 'olive', 'navy', 'crimson', 'scarlet',
    # --- Additional verbs / gerunds ---
    'unhang', 'flick', 'create', 'stretch', 'cooking', 'washing',
    'drying', 'masking', 'measuring', 'charging', 'chopping', 'filing',
    'shaving', 'building', 'polish',
}


def extract_droid_object_vocabulary(
    droid_tasks: List[str] = None,
    droid_episodes: List[List[str]] = None,
    save_path: Optional[str] = None,
) -> List[tuple]:
    """Extract candidate object nouns from DROID instructions.

    Accepts either a flat list of task strings (from tasks.jsonl) or a list
    of episodes where each episode has multiple annotation strings (from
    language_annotations.json).  Each word is counted once per episode/task.
    """
    freq = Counter()

    if droid_episodes is not None:
        total = len(droid_episodes)
        for instructions in droid_episodes:
            episode_words = set()
            for text in instructions:
                episode_words.update(tokenize_instruction(text))
            freq.update(episode_words - NON_OBJECT_WORDS)
    elif droid_tasks is not None:
        total = len(droid_tasks)
        for task in droid_tasks:
            words = set(tokenize_instruction(task))
            freq.update(words - NON_OBJECT_WORDS)
    else:
        return []

    ranked = freq.most_common()

    if save_path:
        import csv
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['object', 'task_count', 'pct_of_tasks'])
            for word, count in ranked:
                writer.writerow([word, count, f'{100.0 * count / total:.1f}%'])
        print(f"Saved {len(ranked)} candidate object words to {save_path}")

    return ranked


def normalize_benchmark_objects_to_words(tasks_data: List[Dict]) -> set:
    """Extract individual object words from benchmark contact_objects.

    Splits compound names on underscores, strips numeric suffixes and
    color/material/size adjectives to yield base object nouns.
    """
    words = set()
    for task in tasks_data:
        objects_str = task.get('contact_objects', '')
        if not objects_str:
            continue
        for obj in objects_str.split(','):
            obj = obj.strip().lower()
            if not obj or obj == 'table':
                continue
            name = re.sub(r'_?\d+$', '', obj)
            name = re.sub(r'_[a-z]\d+$', '', name)
            for part in name.split('_'):
                if part and part not in NON_OBJECT_WORDS and part not in STOPWORDS and len(part) > 1:
                    words.add(part)
    return words


def compute_word_level_object_overlap(
    tasks_data: List[Dict],
    droid_obj_vocab: List[tuple],
    top_pct: float = 1.0,
) -> Dict[str, Any]:
    """Compute word-level overlap between benchmark object words and DROID
    extracted object vocabulary.

    Args:
        top_pct: Fraction of the ranked DROID vocabulary to keep (0.5 = top 50%).
    """
    cutoff = max(1, int(len(droid_obj_vocab) * top_pct))
    droid_obj_vocab = droid_obj_vocab[:cutoff]
    droid_words = {word for word, _ in droid_obj_vocab}
    bench_words = normalize_benchmark_objects_to_words(tasks_data)

    overlap = droid_words & bench_words
    droid_only = droid_words - bench_words
    bench_only = bench_words - droid_words
    all_words = droid_words | bench_words
    jaccard = len(overlap) / len(all_words) if all_words else 0
    droid_coverage = len(overlap) / len(droid_words) if droid_words else 0
    bench_coverage = len(overlap) / len(bench_words) if bench_words else 0

    droid_freq = {word: count for word, count in droid_obj_vocab}

    return {
        'overlap': sorted(overlap),
        'droid_only': sorted(droid_only),
        'bench_only': sorted(bench_only),
        'num_droid': len(droid_words),
        'num_bench': len(bench_words),
        'top_pct': top_pct,
        'jaccard': jaccard,
        'droid_coverage': droid_coverage,
        'bench_coverage': bench_coverage,
        'droid_freq': droid_freq,
    }


def plot_word_level_object_venn(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    setup_plot_style()

    droid_only = results['droid_only']
    bench_only = results['bench_only']
    overlap = results['overlap']
    droid_freq = results['droid_freq']

    overlap_by_freq = sorted(overlap, key=lambda w: -droid_freq.get(w, 0))

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.2)

    # --- Top: Venn diagram ---
    ax_venn = fig.add_subplot(gs[0])

    v = venn2(
        subsets=(len(droid_only), len(bench_only), len(overlap)),
        set_labels=('', ''),
        ax=ax_venn,
    )

    patch_styles = {
        '10': (COLORS['droid'], 0.40),
        '01': (COLORS['benchmark'], 0.40),
        '11': (COLORS['overlap'], 0.55),
    }
    for pid, (color, alpha) in patch_styles.items():
        p = v.get_patch_by_id(pid)
        if p:
            p.set_color(color)
            p.set_alpha(alpha)
            p.set_edgecolor('white')

    venn2_circles(
        subsets=(len(droid_only), len(bench_only), len(overlap)),
        ax=ax_venn, linewidth=2, color='#444444',
    )

    for pid, count in [('10', len(droid_only)),
                       ('01', len(bench_only)),
                       ('11', len(overlap))]:
        lbl = v.get_label_by_id(pid)
        if lbl:
            lbl.set_text(str(count))
            lbl.set_fontsize(22)
            lbl.set_fontweight('bold')
            lbl.set_color('#222222')

    ax_venn.text(0.20, 0.90,
                 f'DROID vocabulary\n({results["num_droid"]:,} words)',
                 ha='center', fontsize=14, fontweight='bold',
                 color=COLORS['droid'], transform=ax_venn.transAxes)
    ax_venn.text(0.80, 0.90,
                 f'Benchmark objects\n({results["num_bench"]} words)',
                 ha='center', fontsize=14, fontweight='bold',
                 color=COLORS['benchmark'], transform=ax_venn.transAxes)

    stats = (f"Jaccard: {results['jaccard']:.3f}        "
             f"Benchmark coverage: {results['bench_coverage']:.1%}        "
             f"DROID coverage: {results['droid_coverage']:.1%}")
    ax_venn.text(0.5, 0.02, stats, ha='center', fontsize=11,
                 style='italic', color='#555555',
                 transform=ax_venn.transAxes)

    pct_label = (f' (top {results["top_pct"]:.0%})'
                 if results.get('top_pct', 1.0) < 1.0 else '')
    ax_venn.set_title(
        f'Word-Level Object Vocabulary Overlap{pct_label}\n'
        '(DROID extracted nouns vs. Benchmark contact objects)',
        fontsize=16, fontweight='bold', pad=15)

    # --- Bottom: three-column word lists ---
    ax_list = fig.add_subplot(gs[1])
    ax_list.axis('off')

    droid_only_by_freq = sorted(droid_only, key=lambda w: -droid_freq.get(w, 0))

    col_x = [0.02, 0.35, 0.70]
    headers = [
        (f'DROID only ({len(droid_only)})', COLORS['droid']),
        (f'Both ({len(overlap)})', COLORS['overlap']),
        (f'Benchmark only ({len(bench_only)})', COLORS['benchmark']),
    ]
    lists = [droid_only_by_freq, overlap_by_freq, bench_only]
    max_show = 30

    for col_idx, ((header, color), items) in enumerate(zip(headers, lists)):
        x = col_x[col_idx]
        ax_list.text(x, 0.98, header, transform=ax_list.transAxes,
                     fontsize=12, fontweight='bold', color=color, va='top')
        ax_list.plot([x, x + 0.26], [0.955, 0.955],
                     transform=ax_list.transAxes, color=color,
                     linewidth=1.5, clip_on=False)

        for i, item in enumerate(items[:max_show]):
            y = 0.94 - i * 0.031
            freq_str = ''
            if col_idx < 2 and item in droid_freq:
                freq_str = f'  ({droid_freq[item]:,})'
            ax_list.text(x + 0.01, y, f'• {item}{freq_str}',
                         transform=ax_list.transAxes,
                         fontsize=8.5, va='top', fontfamily='monospace')
        if len(items) > max_show:
            y = 0.94 - max_show * 0.031
            ax_list.text(x + 0.01, y,
                         f'  ... +{len(items) - max_show} more',
                         transform=ax_list.transAxes,
                         fontsize=8, va='top', color='#888888',
                         style='italic')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_object_word_grid(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Annotated word grid showing every benchmark object word, colored by
    whether it appears in the DROID extracted vocabulary."""
    setup_plot_style()

    overlap = set(results['overlap'])
    bench_only = set(results['bench_only'])
    all_bench = sorted(overlap | bench_only)
    n_total = len(all_bench)
    n_covered = len(overlap)
    n_missing = len(bench_only)
    pct = 100.0 * n_covered / n_total if n_total else 0

    droid_freq = results['droid_freq']
    covered_sorted = sorted(overlap, key=lambda w: -droid_freq.get(w, 0))
    missing_sorted = sorted(bench_only)
    ordered = covered_sorted + missing_sorted

    ncols = 6
    nrows = (len(ordered) + ncols - 1) // ncols

    fig_h = 2.2 + nrows * 0.48
    fig = plt.figure(figsize=(14, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, max(nrows * 0.48, 3)],
                          hspace=0.35)

    # --- Top: coverage bar ---
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.barh(0, pct, height=0.6, color=COLORS['overlap'],
                edgecolor='white', linewidth=0.5, label='In DROID')
    ax_bar.barh(0, 100 - pct, left=pct, height=0.6,
                color=COLORS['benchmark_only'], edgecolor='white',
                linewidth=0.5, label='Not in DROID')
    ax_bar.set_xlim(0, 100)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Percentage of benchmark object words', fontsize=11)
    ax_bar.text(pct / 2, 0, f'{n_covered} words\n({pct:.0f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')
    ax_bar.text(pct + (100 - pct) / 2, 0,
                f'{n_missing} words\n({100 - pct:.0f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='#333333')
    ax_bar.legend(loc='upper right', fontsize=10, framealpha=0.9)
    pct_label = (f' (top {results["top_pct"]:.0%})'
                 if results.get('top_pct', 1.0) < 1.0 else '')
    ax_bar.set_title(
        f'Benchmark Object Vocabulary Coverage by DROID{pct_label}\n'
        f'({n_covered} of {n_total} object words found in DROID instructions)',
        fontsize=14, fontweight='bold', pad=10)
    ax_bar.spines['left'].set_visible(False)

    # --- Bottom: word grid ---
    ax_grid = fig.add_subplot(gs[1])
    ax_grid.axis('off')
    ax_grid.set_xlim(0, 1)
    ax_grid.set_ylim(0, 1)

    cell_w = 1.0 / ncols
    cell_h = 1.0 / nrows if nrows > 0 else 1.0

    for idx, word in enumerate(ordered):
        row = idx // ncols
        col = idx % ncols

        x = col * cell_w
        y = 1.0 - (row + 1) * cell_h

        in_droid = word in overlap
        bg_color = COLORS['overlap'] if in_droid else COLORS['benchmark_only']
        bg_alpha = 0.20 if in_droid else 0.25
        text_color = '#1a5276' if in_droid else '#7b4a1e'

        rect = plt.Rectangle(
            (x + 0.005, y + 0.02), cell_w - 0.01, cell_h - 0.04,
            facecolor=bg_color, alpha=bg_alpha, edgecolor=bg_color,
            linewidth=1.0, transform=ax_grid.transAxes, clip_on=False,
        )
        ax_grid.add_patch(rect)

        label = word
        if in_droid and word in droid_freq:
            label = f'{word}  ({droid_freq[word]:,})'

        ax_grid.text(
            x + cell_w / 2, y + cell_h / 2, label,
            ha='center', va='center', fontsize=8.5,
            fontweight='bold' if in_droid else 'normal',
            color=text_color, transform=ax_grid.transAxes,
        )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
# Analysis 4: Instruction Word Overlap (with raw DROID)
########################################################

def compute_word_overlap(tasks_data: List[Dict],
                         droid_tasks: List[str]) -> Dict[str, Any]:
    benchmark_words = set()
    benchmark_word_freq = Counter()
    for task in tasks_data:
        words = set(tokenize_instruction(task.get('instruction', '')))
        benchmark_words.update(words)
        benchmark_word_freq.update(words)

    droid_word_freq = compute_droid_word_frequencies(droid_tasks)
    droid_words = set(droid_word_freq.keys())

    overlap = benchmark_words & droid_words
    benchmark_only = benchmark_words - droid_words
    droid_only = droid_words - benchmark_words

    union = benchmark_words | droid_words
    jaccard = len(overlap) / len(union) if union else 0

    overlap_by_droid_freq = sorted(
        [(w, droid_word_freq[w], benchmark_word_freq.get(w, 0)) for w in overlap],
        key=lambda x: -x[1],
    )

    return {
        'overlap': sorted(overlap),
        'benchmark_only': sorted(benchmark_only),
        'droid_only_count': len(droid_only),
        'jaccard': jaccard,
        'num_benchmark_words': len(benchmark_words),
        'num_droid_words': len(droid_words),
        'num_overlap': len(overlap),
        'overlap_by_droid_freq': overlap_by_droid_freq,
        'droid_word_freq': droid_word_freq,
        'benchmark_word_freq': benchmark_word_freq,
        'total_droid_tasks': len(droid_tasks),
    }


def print_word_overlap(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("ANALYSIS 4: INSTRUCTION WORD OVERLAP (vs raw DROID tasks)")
    print("=" * 80)
    print(f"Benchmark unique content words: {results['num_benchmark_words']}")
    print(f"DROID unique content words:     {results['num_droid_words']}")
    print(f"Overlapping words:              {results['num_overlap']}")
    print(f"Benchmark-only words:           {len(results['benchmark_only'])}")
    print(f"DROID-only words:               {results['droid_only_count']}")
    print(f"Jaccard similarity:             {results['jaccard']:.3f}")
    print(f"Benchmark coverage:             "
          f"{results['num_overlap'] / results['num_benchmark_words']:.1%} "
          f"of benchmark words appear in DROID")

    print(f"\nTop 30 overlapping words (by DROID frequency):")
    print(f"  {'Word':<20s} {'DROID tasks':>12s} {'Bench tasks':>12s} "
          f"{'DROID %':>8s}")
    print("  " + "-" * 55)
    for word, droid_cnt, bench_cnt in results['overlap_by_droid_freq'][:30]:
        pct = droid_cnt / results['total_droid_tasks'] * 100
        print(f"  {word:<20s} {droid_cnt:>12,d} {bench_cnt:>12d} "
              f"{pct:>7.1f}%")

    print(f"\nBenchmark-only words (not in any DROID task):")
    bonly = results['benchmark_only']
    print(f"  {', '.join(bonly)}")


def plot_word_overlap(results: Dict[str, Any],
                      save_path: Optional[str] = None):
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Left panel: top 30 overlapping words with dual bars
    top_n = 30
    top_words = results['overlap_by_droid_freq'][:top_n]
    words = [w for w, _, _ in top_words]
    droid_counts = [d for _, d, _ in top_words]
    bench_counts = [b for _, _, b in top_words]

    words_rev = list(reversed(words))
    droid_rev = list(reversed(droid_counts))
    bench_rev = list(reversed(bench_counts))

    y = np.arange(len(words_rev))
    height = 0.38

    ax1.barh(y + height / 2, droid_rev, height, color=COLORS['droid'],
             label='DROID tasks', edgecolor='white', linewidth=0.5)

    # Scale benchmark counts to be visible alongside DROID
    max_droid = max(droid_rev) if droid_rev else 1
    max_bench = max(bench_rev) if bench_rev else 1
    scale = max_droid / max_bench * 0.3
    bench_scaled = [b * scale for b in bench_rev]
    ax1.barh(y - height / 2, bench_scaled, height, color=COLORS['benchmark'],
             label=f'Benchmark tasks (×{scale:.0f} scaled)', edgecolor='white',
             linewidth=0.5)

    ax1.set_yticks(y)
    ax1.set_yticklabels(words_rev, fontsize=10)
    ax1.set_xlabel('Task Count')
    ax1.set_title(f'Top {top_n} Shared Words by DROID Frequency',
                   fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10, loc='lower right')

    for i, (dv, bv, br) in enumerate(zip(droid_rev, bench_rev, bench_scaled)):
        ax1.text(dv + max_droid * 0.01, i + height / 2, f'{dv:,}',
                 va='center', fontsize=7, color=COLORS['droid'])
        ax1.text(br + max_droid * 0.01, i - height / 2, f'{bv}',
                 va='center', fontsize=7, color=COLORS['benchmark'])

    # Right panel: summary
    ax2.axis('off')
    total = results['total_droid_tasks']
    bw = results['num_benchmark_words']
    coverage = results['num_overlap'] / bw if bw else 0

    lines = [
        "Instruction Word Overlap",
        "─" * 32, "",
        f"DROID tasks:        {total:,}",
        f"DROID vocab:        {results['num_droid_words']:,}",
        f"Benchmark vocab:    {bw}", "",
        f"Shared words:       {results['num_overlap']}",
        f"Jaccard:            {results['jaccard']:.3f}",
        f"Bench coverage:     {coverage:.1%}", "",
        "─" * 32,
        f"Benchmark-only ({len(results['benchmark_only'])}):",
    ]
    for w in results['benchmark_only']:
        lines.append(f"  • {w}")

    ax2.text(0.05, 0.95, "\n".join(lines), transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                       edgecolor='#DDDDDD'))

    plt.suptitle('Instruction Word Overlap: Benchmark vs DROID',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
# Analysis 5: Per-Task DROID Word Similarity
########################################################

def compute_per_task_word_similarity(tasks_data: List[Dict],
                                     droid_tasks: List[str]) -> List[Dict]:
    droid_word_freq = compute_droid_word_frequencies(droid_tasks)
    total_droid = len(droid_tasks)

    results = []
    for task in tasks_data:
        instruction = task.get('instruction', '')
        words = set(tokenize_instruction(instruction))

        if not words:
            results.append({
                'task_name': task.get('task_name', 'Unknown'),
                'instruction': instruction,
                'word_coverage': 0.0,
                'avg_droid_freq': 0.0,
                'score': 0.0,
                'difficulty_label': task.get('difficulty_label', 'simple'),
                'difficulty_score': task.get('difficulty_score', 0),
                'num_content_words': 0,
                'num_matched': 0,
            })
            continue

        matched = [w for w in words if w in droid_word_freq]
        word_coverage = len(matched) / len(words)

        freqs = [droid_word_freq.get(w, 0) / total_droid for w in words]
        avg_freq = float(np.mean(freqs))

        # Combined score: how many of this task's words appear in DROID,
        # weighted by how common those words are in DROID
        score = 0.6 * word_coverage + 0.4 * avg_freq

        results.append({
            'task_name': task.get('task_name', 'Unknown'),
            'instruction': instruction,
            'word_coverage': word_coverage,
            'avg_droid_freq': avg_freq,
            'score': score,
            'difficulty_label': task.get('difficulty_label', 'simple'),
            'difficulty_score': task.get('difficulty_score', 0),
            'num_content_words': len(words),
            'num_matched': len(matched),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def print_per_task_word_similarity(results: List[Dict]):
    print("\n" + "=" * 80)
    print("ANALYSIS 5: PER-TASK DROID WORD SIMILARITY")
    print("=" * 80)
    print("score = 0.6 * word_coverage + 0.4 * avg_droid_word_freq")
    print("word_coverage = fraction of task's content words found in DROID\n")

    scores = [r['score'] for r in results]
    coverages = [r['word_coverage'] for r in results]
    print(f"Average score:         {np.mean(scores):.3f}")
    print(f"Median score:          {np.median(scores):.3f}")
    print(f"Average word coverage: {np.mean(coverages):.1%}")

    print("\nBy difficulty level:")
    for label in ['simple', 'moderate', 'complex']:
        subset = [r['score'] for r in results if r['difficulty_label'] == label]
        cov = [r['word_coverage'] for r in results if r['difficulty_label'] == label]
        if subset:
            print(f"  {label:10s}: score={np.mean(subset):.3f}  "
                  f"coverage={np.mean(cov):.1%}  n={len(subset)}")

    print(f"\nTop 10 most similar to DROID:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i:2d}. {r['task_name']:<42s} score={r['score']:.3f}  "
              f"coverage={r['word_coverage']:.0%} "
              f"({r['num_matched']}/{r['num_content_words']})  "
              f"[{r['difficulty_label']}]")
        print(f"      \"{r['instruction']}\"")

    print(f"\nBottom 10 least similar to DROID:")
    for i, r in enumerate(results[-10:], 1):
        print(f"  {i:2d}. {r['task_name']:<42s} score={r['score']:.3f}  "
              f"coverage={r['word_coverage']:.0%} "
              f"({r['num_matched']}/{r['num_content_words']})  "
              f"[{r['difficulty_label']}]")
        print(f"      \"{r['instruction']}\"")


def plot_per_task_word_similarity(results: List[Dict],
                                  save_path: Optional[str] = None):
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.5]}
    )

    difficulty_order = ['simple', 'moderate', 'complex']
    difficulty_colors = [COLORS[d] for d in difficulty_order]

    # Left panel: box + strip by difficulty
    data_by_diff = []
    for label in difficulty_order:
        subset = [r['score'] for r in results if r['difficulty_label'] == label]
        data_by_diff.append(subset)

    bp = ax1.boxplot(data_by_diff, vert=True, patch_artist=True,
                     labels=[f"{d}\n(n={len(s)})" for d, s in
                             zip(difficulty_order, data_by_diff)],
                     widths=0.55, showfliers=False)
    for patch, color in zip(bp['boxes'], difficulty_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    for i, (label, subset) in enumerate(zip(difficulty_order, data_by_diff)):
        x = np.random.default_rng(42).normal(i + 1, 0.07, size=len(subset))
        ax1.scatter(x, subset, alpha=0.6, s=30, color=difficulty_colors[i],
                    edgecolor='white', linewidth=0.5, zorder=5)

    ax1.set_ylabel('DROID Word Similarity Score', fontsize=12)
    ax1.set_title('Score by Difficulty Level', fontweight='bold')

    # Right panel: top 10 and bottom 10
    n_show = 10
    top = results[:n_show]
    bottom = results[-n_show:]

    labels_list, scores_list, bar_colors = [], [], []

    for r in reversed(bottom):
        short = r['task_name'].replace('Task', '')
        labels_list.append(short)
        scores_list.append(r['score'])
        bar_colors.append(COLORS[r['difficulty_label']])

    sep_idx = len(labels_list)
    labels_list.append('')
    scores_list.append(0)
    bar_colors.append('white')

    for r in reversed(top):
        short = r['task_name'].replace('Task', '')
        labels_list.append(short)
        scores_list.append(r['score'])
        bar_colors.append(COLORS[r['difficulty_label']])

    y_pos = range(len(labels_list))
    ax2.barh(y_pos, scores_list, color=bar_colors,
             edgecolor='white', linewidth=0.5, height=0.75)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels_list, fontsize=9)
    ax2.set_xlabel('DROID Word Similarity Score')
    ax2.set_title('Highest & Lowest DROID Word Similarity', fontweight='bold')

    for i, (score, label) in enumerate(zip(scores_list, labels_list)):
        if label:
            ax2.text(score + 0.005, i, f'{score:.2f}', va='center', fontsize=8)

    ax2.axhline(y=sep_idx, color='gray', linestyle='--', alpha=0.5)
    ax2.text(max(scores_list) * 0.5, sep_idx,
             '— gap —', ha='center', va='center',
             fontsize=8, color='gray', style='italic',
             bbox=dict(facecolor='white', edgecolor='none', pad=1))

    patches = [mpatches.Patch(color=COLORS[d], label=d) for d in difficulty_order]
    ax2.legend(handles=patches, loc='center right', fontsize=9)

    plt.suptitle('Per-Task DROID Word Similarity (from raw DROID instructions)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


########################################################
# Main
########################################################

def main():
    parser = argparse.ArgumentParser(
        description='Compare robolab benchmark tasks against DROID dataset distribution'
    )
    parser.add_argument('--metadata', default=None,
                        help='Path to task_metadata.json')
    parser.add_argument('--droid-tasks', default=None,
                        help='Path to DROID tasks.jsonl for word-level analysis')
    parser.add_argument('--droid-annotations', default=None,
                        help='Path to droid_language_annotations.json')
    parser.add_argument('--save-dir', default=None,
                        help='Directory to save figures (default: show interactively)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting, print analysis only')
    args = parser.parse_args()

    if args.metadata:
        metadata_path = args.metadata
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        metadata_path = os.path.join(
            project_root, 'robolab', 'tasks', '_metadata', 'task_metadata.json')

    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found: {metadata_path}")
        print("Run generate_task_metadata.py first.")
        return

    with open(metadata_path, 'r') as f:
        tasks_data = json.load(f)
    print(f"Loaded {len(tasks_data)} benchmark tasks from {metadata_path}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # --- Analyses 1-3: hardcoded DROID distribution constants ---
    overlap = compute_object_overlap(tasks_data)
    print_object_overlap(overlap)

    scores = compute_per_task_scores(tasks_data)
    print_per_task_scores(scores)

    complexity = compute_complexity_comparison(tasks_data)
    print_complexity_comparison(complexity)

    # --- Analyses 4-5: raw DROID task instructions ---
    droid_tasks = None
    droid_episodes = None

    # Try language annotations first (richer: 3 annotations per episode)
    annotations_path = args.droid_annotations
    if annotations_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        candidate = os.path.join(
            project_root, 'analysis', 'droid_language_annotations.json')
        if os.path.exists(candidate):
            annotations_path = candidate

    if annotations_path and os.path.exists(annotations_path):
        print(f"\nLoading DROID language annotations from {annotations_path}...")
        droid_episodes = load_droid_language_annotations(annotations_path)
        print(f"Loaded {len(droid_episodes)} DROID episodes "
              f"(~{sum(len(e) for e in droid_episodes)} total annotations)")
        droid_tasks = [instr for ep in droid_episodes for instr in ep]

    if droid_tasks is None:
        droid_tasks_path = args.droid_tasks
        if droid_tasks_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            candidate = os.path.join(project_root, 'assets', 'tasks.jsonl')
            if os.path.exists(candidate):
                droid_tasks_path = candidate
        if droid_tasks_path and os.path.exists(droid_tasks_path):
            print(f"\nLoading DROID tasks from {droid_tasks_path}...")
            droid_tasks = load_droid_tasks(droid_tasks_path)
            print(f"Loaded {len(droid_tasks)} DROID task instructions")

    if droid_tasks:
        word_overlap = compute_word_overlap(tasks_data, droid_tasks)
        print_word_overlap(word_overlap)

        word_sim = compute_per_task_word_similarity(tasks_data, droid_tasks)
        print_per_task_word_similarity(word_sim)

        obj_csv = os.path.join(
            args.save_dir or 'analysis', 'droid_object_list.csv')
        n_episodes = len(droid_episodes) if droid_episodes else len(droid_tasks)
        droid_obj_vocab = extract_droid_object_vocabulary(
            droid_tasks=(droid_tasks if droid_episodes is None else None),
            droid_episodes=droid_episodes,
            save_path=obj_csv)
        print(f"\nDROID object vocabulary: {len(droid_obj_vocab)} unique candidate words")
        print("Top 30:")
        for word, count in droid_obj_vocab[:30]:
            pct = 100.0 * count / n_episodes
            print(f"  {word:<25s} {count:>6d} episodes ({pct:5.1f}%)")

        word_obj_overlap = compute_word_level_object_overlap(
            tasks_data, droid_obj_vocab, top_pct=0.5)
        n_cutoff = max(1, int(len(droid_obj_vocab) * 0.5))
        print(f"\n--- Word-Level Object Overlap (top 50% = {n_cutoff} DROID words) ---")
        print(f"Benchmark object words: {word_obj_overlap['num_bench']}")
        print(f"DROID vocabulary words: {word_obj_overlap['num_droid']:,}")
        print(f"Overlap: {len(word_obj_overlap['overlap'])}")
        print(f"Jaccard: {word_obj_overlap['jaccard']:.3f}")
        print(f"Benchmark coverage: {word_obj_overlap['bench_coverage']:.1%}")
        print(f"DROID coverage: {word_obj_overlap['droid_coverage']:.1%}")
        print(f"Benchmark-only: {', '.join(word_obj_overlap['bench_only'])}")
    else:
        word_overlap = None
        word_sim = None
        droid_obj_vocab = None
        word_obj_overlap = None
        print("\nSkipping word-level analyses (no DROID data found).")
        print("Provide --droid-annotations or --droid-tasks to enable.")

    # --- Plots ---
    if not args.no_plot:
        sd = args.save_dir
        print("\nGenerating figures...")
        plot_object_overlap(
            overlap,
            save_path=os.path.join(sd, 'object_overlap.png') if sd else None)
        plot_object_venn(
            overlap,
            save_path=os.path.join(sd, 'object_venn.png') if sd else None)
        plot_per_task_scores(
            scores,
            save_path=os.path.join(sd, 'per_task_scores.png') if sd else None)
        plot_complexity_comparison(
            complexity,
            save_path=os.path.join(sd, 'complexity_comparison.png') if sd else None)

        if word_overlap:
            plot_word_overlap(
                word_overlap,
                save_path=os.path.join(sd, 'word_overlap.png') if sd else None)
        if word_sim:
            plot_per_task_word_similarity(
                word_sim,
                save_path=os.path.join(sd, 'word_similarity.png') if sd else None)
        if word_obj_overlap:
            plot_word_level_object_venn(
                word_obj_overlap,
                save_path=os.path.join(sd, 'object_word_venn.png') if sd else None)
            plot_object_word_grid(
                word_obj_overlap,
                save_path=os.path.join(sd, 'object_word_grid.png') if sd else None)

        if not sd:
            plt.show()

    print("\nDone.")


if __name__ == '__main__':
    main()
