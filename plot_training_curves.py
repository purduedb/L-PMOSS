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

def convert_timestamps_to_minutes(timestamps, gap_threshold_minutes=10):
    """
    Convert timestamps to cumulative minutes elapsed from training start.
    X-axis = total training time up to each checkpoint.
    
    Detects and removes gaps in training (e.g., when training is stopped and restarted).
    If time between consecutive timestamps exceeds gap_threshold_minutes, the gap is removed.
    
    Args:
        timestamps: List of timestamp strings
        gap_threshold_minutes: Maximum expected time between checkpoints (default 10 minutes)
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

    # Calculate cumulative elapsed time, removing gaps
    cumulative_minutes = [0.0]  # First timestamp is t=0
    total_gap_removed = 0.0  # Track total gap time removed
    
    for i in range(1, len(datetime_objs)):
        # Calculate actual elapsed time since previous timestamp
        elapsed_seconds = (datetime_objs[i] - datetime_objs[i-1]).total_seconds()
        elapsed_minutes = elapsed_seconds / 60.0
        
        # Detect gap: if elapsed time is unusually large, it's likely a restart
        if elapsed_minutes > gap_threshold_minutes:
            # This is a gap - calculate what the "normal" increment should be
            # Use the average of recent increments (last 5 checkpoints or available)
            recent_increments = []
            for j in range(max(0, i-5), i):
                if j > 0:
                    recent_inc = cumulative_minutes[j] - cumulative_minutes[j-1]
                    if recent_inc > 0 and recent_inc <= gap_threshold_minutes:
                        recent_increments.append(recent_inc)
            
            # Use average recent increment, or a default of 2 minutes if no history
            if recent_increments:
                expected_increment = sum(recent_increments) / len(recent_increments)
            else:
                expected_increment = 2.0  # Default assumption: 2 minutes per checkpoint
            
            # Track the gap we're removing
            gap_size = elapsed_minutes - expected_increment
            total_gap_removed += gap_size
            
            print(f"  → Gap detected at timestamp {i}: {elapsed_minutes:.1f} min gap, "
                  f"normalizing to {expected_increment:.1f} min (removed {gap_size:.1f} min)")
            
            # Use expected increment instead of actual
            cumulative_minutes.append(cumulative_minutes[-1] + expected_increment)
        else:
            # Normal increment - add to cumulative time
            cumulative_minutes.append(cumulative_minutes[-1] + elapsed_minutes)
    
    if total_gap_removed > 0:
        print(f"  → Total gap time removed: {total_gap_removed:.1f} minutes ({total_gap_removed/60:.1f} hours)")
    
    return cumulative_minutes

# Parse all three log files
print("="*70)
print("Parsing log files...")
print("="*70)

print("\n[1/3] Parsing BC baseline log...")
bc_epochs, bc_accuracies, bc_timestamps = parse_log_with_timestamps(
    # '/home/yrayhan/works/L-PMOSS/_log_bc_baseline.txt', 'BC'
    '/home/yrayhan/works/L-PMOSS/__log_bc_baseline.txt', 'BC'
    )

print("[3/3] Parsing CQL baseline log...")
cql_epochs, cql_accuracies, cql_timestamps = parse_log_with_timestamps(
    # '/home/yrayhan/works/L-PMOSS/_log_bc_baseline.txt', 'BC'
    # '/home/yrayhan/works/L-PMOSS/log_cql_baseline.txt', 'CQL'
    '/home/yrayhan/works/L-PMOSS/___log_cql_baseline.txt', 'CQL'
    )

print("[2/3] Parsing DT baseline log...")
dt_epochs, dt_accuracies, dt_timestamps = parse_log_with_timestamps(
    '/home/yrayhan/works/L-PMOSS/log_dt_baseline.txt', 'DT'
    # '/home/yrayhan/works/L-PMOSS/log_cql_baseline.txt', 'BC'
    )

cql_epochs2, cql_accuracies2, cql_timestamps2 = parse_log_with_timestamps(
    # '/home/yrayhan/works/L-PMOSS/__log_cql_baseline.txt', 'CQL'
    '/home/yrayhan/works/L-PMOSS/____log_cql_baseline.txt', 'CQL'
    )

# Convert timestamps to cumulative minutes from start
bc_minutes = convert_timestamps_to_minutes(bc_timestamps)
dt_minutes = convert_timestamps_to_minutes(dt_timestamps)
cql_minutes = convert_timestamps_to_minutes(cql_timestamps)
cql_minutes2 = convert_timestamps_to_minutes(cql_timestamps2)

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

fig, ax = plt.subplots(figsize=(6, 3))
# fig, ax = plt.subplots(figsize=(6, 6))

# Plot all three curves
linewidth=1
if bc_minutes and bc_accuracies:
    ax.plot(bc_minutes, bc_accuracies, 'b-', label='BC',
            linewidth=linewidth, marker='o', markersize=4, markevery=max(1, len(bc_minutes)//30), alpha=0.9)

if dt_minutes and dt_accuracies:
    ax.plot(dt_minutes, dt_accuracies, 'r-', label='DT',
            linewidth=linewidth, marker='s', markersize=4, markevery=max(1, len(dt_minutes)//30), alpha=0.9)

if cql_minutes and cql_accuracies:
    ax.plot(cql_minutes, cql_accuracies, 'g-', label='CQL',
            linewidth=linewidth, marker='^', markersize=4, markevery=max(1, len(cql_minutes)//30), alpha=0.9)

# make color orange for cql_minutes2
if cql_minutes2 and cql_accuracies2:
    ax.plot(cql_minutes2, cql_accuracies2, 'orange', label='CQL (v2)',
            linewidth=linewidth, marker='o', markersize=4, markevery=max(1, len(cql_minutes2)//30), alpha=0.9)

# Formatting
ax.set_xlabel('Training Time (minutes)', fontsize=16)
ax.set_ylabel('Accuracy (%)', fontsize=16)
# ax.set_title('Training Convergence Comparison: BC vs DT vs CQL', fontsize=18)
# How to turn off the edge in legend box?
ax.legend(fontsize=10, loc='lower right', edgecolor='none', ncols=4)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.tick_params(axis='both', which='major', labelsize=12)

# Set reasonable limits
if bc_accuracies or dt_accuracies or cql_accuracies:
    all_accs = (bc_accuracies or []) + (dt_accuracies or []) + (cql_accuracies or [])
    ax.set_ylim([0, min(100, max(all_accs) * 1.07)])

plt.tight_layout()

# Save the main plot
output_file = '/home/yrayhan/works/L-PMOSS/training_curves_time.png'
plt.savefig(output_file, dpi=100, bbox_inches='tight')
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
