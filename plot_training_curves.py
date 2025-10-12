#!/usr/bin/env python3
"""
Plot training accuracy over time for BC, CQL, and DT models.
Extracts timestamps and action_match (accuracy) from log files.
X-axis shows cumulative training time (elapsed time from start).
"""

import re
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def parse_log_with_timestamps(filepath, algo_name):
    """
    Parse log file with format:
    BC/CQL: [2m2025-10-11 23:38.46[0m ... action_match': 0.096...
    DT: Epoch: 1, Time: 2025-10-11-01-30-41, Accuracy: 0.0350
    """
    epochs = []
    accuracies = []
    timestamps = []

    # Pattern to match the log lines
    if algo_name == "DT":
        # DT has different format: Epoch: 1, Time: 2025-10-11-01-30-41, Accuracy: 0.0350
        pattern = r'Epoch: (\d+), Time: ([\d-]+), Accuracy: ([\d.]+)'
    else:
        # BC and CQL format: [2m2025-10-11 23:38.46[0m ... action_match': 0.096...
        # Simplified pattern - match timestamp, epoch, and action_match value
        pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}\.\d{2}).*?epoch=(\d+).*?action_match.*?:\s*([\d.]+)'

    with open(filepath, 'r') as f:
        for line in f:
            if algo_name == "DT":
                match = re.search(pattern, line)
                if match:
                    epoch = int(match.group(1))
                    timestamp_str = match.group(2)
                    accuracy = float(match.group(3)) * 100  # Convert to percentage

                    epochs.append(epoch)
                    accuracies.append(accuracy)
                    timestamps.append(timestamp_str)
            else:
                match = re.search(pattern, line)
                if match:
                    date_str = match.group(1)
                    time_str = match.group(2)
                    epoch = int(match.group(3))
                    accuracy = float(match.group(4)) * 100  # Convert to percentage

                    # Combine date and time
                    timestamp_str = f"{date_str} {time_str}"

                    epochs.append(epoch)
                    accuracies.append(accuracy)
                    timestamps.append(timestamp_str)

    return epochs, accuracies, timestamps

def convert_timestamps_to_minutes(timestamps):
    """
    Convert timestamps to cumulative minutes elapsed from training start.
    X-axis = total training time up to each checkpoint.
    """
    if not timestamps:
        return []

    datetime_objs = []
    for ts in timestamps:
        try:
            if ' ' in ts:
                # Format: "2025-10-11 23:38.46"
                dt = datetime.strptime(ts, '%Y-%m-%d %H:%M.%S')
            else:
                # Format: "2025-10-11-01-30-41"
                dt = datetime.strptime(ts, '%Y-%m-%d-%H-%M-%S')
            datetime_objs.append(dt)
        except Exception as e:
            print(f"Warning: Could not parse timestamp '{ts}': {e}")
            continue

    if not datetime_objs:
        return []

    # First timestamp is t=0 (training starts)
    start_time = datetime_objs[0]

    # Calculate cumulative elapsed time from start
    cumulative_minutes = []
    for dt in datetime_objs:
        elapsed_seconds = (dt - start_time).total_seconds()
        elapsed_minutes = elapsed_seconds / 60.0
        cumulative_minutes.append(elapsed_minutes)

    return cumulative_minutes

# Parse all three log files
print("="*70)
print("Parsing log files...")
print("="*70)

print("\n[1/3] Parsing BC baseline log...")
bc_epochs, bc_accuracies, bc_timestamps = parse_log_with_timestamps(
    '/home/yrayhan/works/L-PMOSS/log_bc_baseline.txt', 'BC')

print("[2/3] Parsing DT baseline log...")
dt_epochs, dt_accuracies, dt_timestamps = parse_log_with_timestamps(
    '/home/yrayhan/works/L-PMOSS/log_dt_baseline.txt', 'DT')

print("[3/3] Parsing CQL baseline log...")
cql_epochs, cql_accuracies, cql_timestamps = parse_log_with_timestamps(
    '/home/yrayhan/works/L-PMOSS/log_cql_baseline.txt', 'CQL')

# Convert timestamps to cumulative minutes from start
bc_minutes = convert_timestamps_to_minutes(bc_timestamps)
dt_minutes = convert_timestamps_to_minutes(dt_timestamps)
cql_minutes = convert_timestamps_to_minutes(cql_timestamps)

# Print statistics
print("\n" + "="*70)
print("Data Summary:")
print("="*70)
if bc_epochs:
    print(f"BC:  {len(bc_epochs):4d} checkpoints | Max Epoch: {max(bc_epochs):7d} | "
          f"Final Accuracy: {bc_accuracies[-1]:5.2f}% | Training Time: {bc_minutes[-1]:7.1f} min")
if dt_epochs:
    print(f"DT:  {len(dt_epochs):4d} checkpoints | Max Epoch: {max(dt_epochs):7d} | "
          f"Final Accuracy: {dt_accuracies[-1]:5.2f}% | Training Time: {dt_minutes[-1]:7.1f} min")
if cql_epochs:
    print(f"CQL: {len(cql_epochs):4d} checkpoints | Max Epoch: {max(cql_epochs):7d} | "
          f"Final Accuracy: {cql_accuracies[-1]:5.2f}% | Training Time: {cql_minutes[-1]:7.1f} min")

# Create the main plot: Cumulative Time vs Accuracy
print("\n" + "="*70)
print("Creating plots...")
print("="*70)

fig, ax = plt.subplots(figsize=(3, 3))

# Plot all three curves
if bc_minutes and bc_accuracies:
    ax.plot(bc_minutes, bc_accuracies, 'b-', label='BC',
            linewidth=2.5, marker='o', markersize=4, markevery=max(1, len(bc_minutes)//30), alpha=0.9)

if dt_minutes and dt_accuracies:
    ax.plot(dt_minutes, dt_accuracies, 'r-', label='DT',
            linewidth=2.5, marker='s', markersize=4, markevery=max(1, len(dt_minutes)//30), alpha=0.9)

if cql_minutes and cql_accuracies:
    ax.plot(cql_minutes, cql_accuracies, 'g-', label='CQL',
            linewidth=2.5, marker='^', markersize=4, markevery=max(1, len(cql_minutes)//30), alpha=0.9)

# Formatting
ax.set_xlabel('Training Time (minutes)', fontsize=16)
ax.set_ylabel('Accuracy (%)', fontsize=16)
# ax.set_title('Training Convergence Comparison: BC vs DT vs CQL', fontsize=18)
# How to turn off the edge in legend box?
ax.legend(fontsize=12, loc='lower right', framealpha=0.95, edgecolor='none')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.tick_params(axis='both', which='major', labelsize=12)

# Set reasonable limits
if bc_accuracies or dt_accuracies or cql_accuracies:
    all_accs = (bc_accuracies or []) + (dt_accuracies or []) + (cql_accuracies or [])
    ax.set_ylim([0, min(100, max(all_accs) * 1.05)])

plt.tight_layout()

# Save the main plot
output_file = '/home/yrayhan/works/L-PMOSS/training_curves_time.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
output_file = '/home/yrayhan/works/L-PMOSS/training_curves_time.pdf'
plt.savefig(output_file, bbox_inches='tight', format='pdf')
print(f"\n✓ Time-based plot saved to: {output_file}")

# Create secondary plot: Epochs vs Accuracy
# fig2, ax2 = plt.subplots(figsize=(14, 8))

# if bc_epochs and bc_accuracies:
#     ax2.plot(bc_epochs, bc_accuracies, 'b-', label='BC (Behavior Cloning)',
#              linewidth=2.5, marker='o', markersize=4, markevery=max(1, len(bc_epochs)//30), alpha=0.9)

# if dt_epochs and dt_accuracies:
#     ax2.plot(dt_epochs, dt_accuracies, 'r-', label='DT (Decision Transformer)',
#              linewidth=2.5, marker='s', markersize=4, markevery=max(1, len(dt_epochs)//30), alpha=0.9)

# if cql_epochs and cql_accuracies:
#     ax2.plot(cql_epochs, cql_accuracies, 'g-', label='CQL (Conservative Q-Learning)',
#              linewidth=2.5, marker='^', markersize=4, markevery=max(1, len(cql_epochs)//30), alpha=0.9)

# ax2.set_xlabel('Training Epochs', fontsize=16, fontweight='bold')
# ax2.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
# ax2.set_title('Training Convergence Comparison: BC vs DT vs CQL', fontsize=18, fontweight='bold', pad=20)
# ax2.legend(fontsize=14, loc='lower right', framealpha=0.95, edgecolor='black')
# ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
# ax2.tick_params(axis='both', which='major', labelsize=12)

# plt.tight_layout()

# output_file_epochs = '/home/yrayhan/works/L-PMOSS/training_curves_epochs.png'
# plt.savefig(output_file_epochs, dpi=300, bbox_inches='tight')
# print(f"✓ Epoch-based plot saved to: {output_file_epochs}")

print("\n" + "="*70)
print("Done! Plots generated successfully.")
print("="*70)
print("\nNote: X-axis shows cumulative training time from start.")
print("      Each point represents total time elapsed up to that checkpoint.")
