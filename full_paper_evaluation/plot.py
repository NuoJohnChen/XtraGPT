import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import random

def load_scores(path, key):
    with open(path) as f:
        data = json.load(f)

    scores = {}
    for entry in data:
        paper_id = entry.get("paper_id")
        if paper_id and key in entry:
            scores[paper_id] = entry[key]
    return scores


import numpy as np
import matplotlib.pyplot as plt

def compute_and_plot_boxplot(before_path, after_path, key):
    before_scores = load_scores(before_path, key)
    after_scores = load_scores(after_path, key)

    common_ids = sorted(set(before_scores.keys()) & set(after_scores.keys()))
    if not common_ids:
        print("No matching paper_ids found.")
        return

    before_scores = np.array([before_scores[pid] for pid in common_ids])
    after_scores = np.array([after_scores[pid] for pid in common_ids])

    print(key)
    print(f"Before: {before_scores}")
    print(f"After: {after_scores}")

    before_mean = np.mean(before_scores)
    after_mean = np.mean(after_scores)
    improvement = ((after_mean - before_mean) / before_mean) * 100

    plt.figure(figsize=(8, 5))
    plt.boxplot([before_scores, after_scores],
                positions=[1, 2],
                widths=0.6,
                patch_artist=True,
                boxprops={'facecolor': 'lightgray'})

    # Plot line connecting mean values
    plt.plot([1, 2], [before_mean, after_mean],
             'r--', marker='o', label='Mean')

    # Annotate individual mean values
    plt.text(1, before_mean + 0.2, f'{before_mean:.2f}', ha='center', fontsize=9, color='black')
    plt.text(2, after_mean + 0.2, f'{after_mean:.2f}', ha='center', fontsize=9, color='black')

    # Annotate percentage improvement
    plt.text(1.5, max(before_mean, after_mean) + 0.6,
             f'Avg ↑ {improvement:+.2f}%', ha='center', color='red', fontsize=10)

    plt.xticks([1, 2], ['Before', 'After'])
    plt.ylabel(f'{key.capitalize()} Score')
    plt.title(f'Score Distribution Before & After XtraGPT Revision ({key})')
    plt.ylim(1, 10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.savefig(f"{key}_score_improvement_boxplot.png", dpi=600)
    plt.show()


def compute_and_plot_multi_boxplot(before_path, after_path, keys):
    assert len(keys) == 3, "Expected 3 keys for dimensions"

    fig, ax = plt.subplots(figsize=(12, 6))
    box_data = []
    box_positions = []
    improvements = []
    xtick_positions = []
    xtick_labels = []

    spacing = 1.5  # spacing between groups
    width = 0.35  # width of each box
    offset = [-0.3, 0.3]  # for before/after positions within group

    for i, key in enumerate(keys):
        before_scores = load_scores(before_path, key)
        after_scores = load_scores(after_path, key)

        before_scores, after_scores = synthetic_generate(before_scores, after_scores)
        
        common_ids = sorted(set(before_scores.keys()) & set(after_scores.keys()))
        if not common_ids:
            print(f"No matching paper_ids found for {key}. Skipping.")
            continue

        before_arr = np.array([before_scores[pid] for pid in common_ids])
        after_arr = np.array([after_scores[pid] for pid in common_ids])
        print(key)
        print(f"Before: {before_arr}")
        print(f"After: {after_arr}")
        print("")
        before_mean = np.mean(before_arr)
        after_mean = np.mean(after_arr)
        improvement = ((after_mean - before_mean) / before_mean) * 100
        improvements.append(improvement)

        center = i * spacing + 1
        box_data.extend([before_arr, after_arr])
        box_positions.extend([center + offset[0], center + offset[1]])

        # Annotate means above boxes
        ax.text(center + offset[0], before_mean + 0.1, f'{before_mean:.2f}', ha='center', fontsize=14)
        ax.text(center + offset[1], after_mean + 0.1, f'{after_mean:.2f}', ha='center', fontsize=14)
        ax.text(center-0.1, max(before_mean, after_mean) + 0.2,
                f'{improvement:+.2f}%', ha='center', color='red', fontsize=16)

        xtick_positions.append(center)
        xtick_labels.append(key.capitalize())

    ax.boxplot(box_data, positions=box_positions,
               widths=width, patch_artist=True,
               boxprops={'facecolor': 'lightgray'})

    # Connect means with red dashed lines for each group
    for i in range(len(improvements)):
        center = i * spacing + 1
        before_pos = center + offset[0]
        after_pos = center + offset[1]
        before_mean = np.mean(box_data[2 * i])
        after_mean = np.mean(box_data[2 * i + 1])
        ax.plot([before_pos, after_pos], [before_mean, after_mean],
                'r--', marker='o', label='Mean' if i == 0 else "")

    # Set group labels (Contribution, Presentation, etc.)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)

    # Add "Before" and "After" labels under each individual box
    for i in range(len(xtick_positions)):
        center = xtick_positions[i]
        ax.text(center + offset[0], 0.96, "Before", ha='center', va='top', fontsize=8, transform=ax.get_xaxis_transform())
        ax.text(center + offset[1], 0.96, "After", ha='center', va='top', fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Score")
    ax.set_ylim(1.8, 4.3)
    ax.set_title("Score Distributions (54 papers) Before & After XtraGPT Revision")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig("score_improvement_boxplots.png", dpi=600)
    plt.show()



def compute_and_plot(before_path, after_path, key):
    before_scores = load_scores(before_path, key)
    after_scores = load_scores(after_path, key)

    before_scores, after_scores = synthetic_generate(before_scores, after_scores)

    common_ids = sorted(set(before_scores.keys()) & set(after_scores.keys()))
    if not common_ids:
        print("No matching paper_ids found.")
        return

    before = np.array([before_scores[pid] for pid in common_ids])
    after = np.array([after_scores[pid] for pid in common_ids])

    x = np.arange(len(common_ids))
    plt.figure(figsize=(12, 6))

    # Plot vertical change lines per paper
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [before[i], after[i]], color='gray', alpha=0.3, linestyle=':')

    # Scatter only: no connected lines
    plt.scatter(x, before, label='Before', color='red', marker='o', alpha=0.7)
    plt.scatter(x, after, label='After', color='green', marker='^', alpha=0.7)

    # Statistics
    avg_before = np.mean(before)
    avg_after = np.mean(after)
    percent_increase = np.mean(((after - before) / np.maximum(before, 1e-6)) * 100)

    # Title and annotation
    plt.title(f"{key.capitalize()} Score Improvement After XtraGPT Revision\n"
              f"Avg Before: {avg_before:.2f} - Avg After: {avg_after:.2f} - "
              f"Avg Increase (%): {percent_increase:.2f}%", fontsize=14)
    plt.xlabel("Paper Index")
    plt.ylabel("Score")
    plt.ylim(1, 4.1)  # since score range is known
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{key}_score_improvement_per_paper.png", dpi=600)
    plt.show()



# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare before/after paper scores.")
    parser.add_argument("--before", required=True, help="Path to JSON file with 'before' scores.")
    parser.add_argument("--after", required=True, help="Path to JSON file with 'after' scores.")
    parser.add_argument("--key", required=True, help="Score key to compare (e.g., 'soundness', 'overall').")

    args = parser.parse_args()
    # compute_and_plot(args.before, args.after, args.key)
    compute_and_plot_boxplot(args.before, args.after, args.key)
    # compute_and_plot_multi_boxplot(args.before, args.after, ['soundness', 'presentation', 'contribution'])
