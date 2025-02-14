import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# plotting style
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ------------------------
# HopfieldNetwork Class
# ------------------------
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.W = np.zeros((num_neurons, num_neurons), dtype=float)
    
    def train_patterns_hebbian(self, patterns, partial_update=False, alpha=0.5):
        """
        Train or partially update the Hopfield network using Hebbian weight changes.
        If partial_update=True, weight updates are scaled by alpha.
        """
        for p in patterns:
            p_bipolar = np.where(p > 0, 1, -1)
            delta_W = np.outer(p_bipolar, p_bipolar)
            np.fill_diagonal(delta_W, 0)
            if partial_update:
                self.W += alpha * delta_W
            else:
                self.W += delta_W
        # Ensure symmetry
        self.W = 0.5 * (self.W + self.W.T)
    
    def retrieve_pattern(self, pattern, steps=10, synchronous=True, track_energy=False):
        """
        Retrieve (settle) a pattern from the network. Optionally track the energy trajectory.
        """
        # Convert binary {0,1} -> bipolar {+1,-1} if needed
        if (pattern == 0).any() or (pattern == 1).any():
            state = np.where(pattern > 0, 1, -1)
        else:
            state = pattern.copy()
        
        energy_list = []
        for _ in range(steps):
            if track_energy:
                energy_list.append(self.energy(state))
            
            if synchronous:
                s = self.W @ state
                state = np.where(s >= 0, 1, -1)
            else:
                for i in np.random.permutation(len(state)):
                    activation = np.dot(self.W[i, :], state)
                    state[i] = 1 if activation >= 0 else -1
        
        if track_energy:
            energy_list.append(self.energy(state))
        
        return state, (energy_list if track_energy else None)
    
    def energy(self, pattern):
        """
        Hopfield energy function: E = -0.5 * x^T W x (for bipolar pattern x).
        """
        return -0.5 * pattern @ self.W @ pattern

# ------------------------
# Helper Functions
# ------------------------
def generate_random_patterns(num_patterns, pattern_size, flip_probability=0.0, base_patterns=None):
    """
    Generate random binary patterns. Optionally flip bits from base_patterns by flip_probability.
    """
    patterns = []
    if base_patterns is None:
        # Fresh random patterns
        for _ in range(num_patterns):
            patterns.append(np.random.choice([0, 1], size=pattern_size))
    else:
        # Flip bits in base_patterns
        for bp in base_patterns:
            new_pattern = bp.copy()
            flips = np.random.rand(pattern_size) < flip_probability
            new_pattern[flips] = 1 - new_pattern[flips]
            patterns.append(new_pattern)
    return np.array(patterns)

def measure_recall_accuracy(hnet, patterns, steps=10, synchronous=True, noise_fraction=0.0):
    """
    For each pattern, optionally flip some fraction of bits (noise_fraction),
    retrieve it, and compute bitwise accuracy.
    """
    accuracies = []
    for p in patterns:
        # Add noise
        noisy_p = p.copy()
        if noise_fraction > 0:
            flips = np.random.rand(len(p)) < noise_fraction
            noisy_p[flips] = 1 - noisy_p[flips]
        
        final_state, _ = hnet.retrieve_pattern(noisy_p, steps=steps, synchronous=synchronous)
        retrieved_binary = np.where(final_state > 0, 1, 0)
        accuracy = np.mean(retrieved_binary == p)
        accuracies.append(accuracy)
    return accuracies

def confusion_matrix_retrieval(hnet, all_patterns, steps=10):
    """
    Compute confusion matrix: for each pattern, see which stored pattern it converges to.
    """
    n = len(all_patterns)
    cm = np.zeros((n, n), dtype=int)
    for i, pattern in enumerate(all_patterns):
        final_state, _ = hnet.retrieve_pattern(pattern, steps=steps, synchronous=True)
        final_binary = np.where(final_state > 0, 1, 0)
        
        # Compare to each reference pattern
        distances = []
        for j, ref_pat in enumerate(all_patterns):
            dist = np.sum(final_binary != ref_pat)
            distances.append(dist)
        best_match = np.argmin(distances)
        cm[i, best_match] += 1
    return cm

# ------------------------
# Single Experiment
# ------------------------
def run_experiment_once(
    seed=42,
    num_neurons=40,
    num_original_patterns=6,
    num_modified_patterns=6,
    alpha=None,
    flip_probability=None,
    num_increments=4,
    noise_levels=None,
    track_energy_patterns=None
):
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    np.random.seed(seed)
    
    # Randomize alpha/flip_probability if not provided
    if alpha is None:
        alpha = np.round(0.1 + 0.7 * np.random.rand(), 2)
    if flip_probability is None:
        flip_probability = np.round(0.05 + 0.25 * np.random.rand(), 2)
    
    # 1. Generate original + modified patterns
    original_patterns = generate_random_patterns(num_original_patterns, num_neurons)
    modified_patterns = generate_random_patterns(
        num_modified_patterns, num_neurons, 
        flip_probability=flip_probability,
        base_patterns=original_patterns
    )
    all_patterns = np.vstack([original_patterns, modified_patterns])
    
    # 2. Initialize and train Hopfield
    hnet = HopfieldNetwork(num_neurons)
    # First, full Hebbian on original
    hnet.train_patterns_hebbian(original_patterns, partial_update=False)
    
    # 3. Baseline accuracy for original
    baseline_acc_original = measure_recall_accuracy(hnet, original_patterns)
    
    # 4. Incremental partial updates with modified patterns
    incremental_acc_original = []
    incremental_acc_modified = []
    
    
    for inc in range(num_increments):
        # partial update
        hnet.train_patterns_hebbian(modified_patterns, partial_update=True, alpha=alpha)
        
        # measure recall each time
        acc_orig = measure_recall_accuracy(hnet, original_patterns)
        acc_mod = measure_recall_accuracy(hnet, modified_patterns)
        incremental_acc_original.append(acc_orig)
        incremental_acc_modified.append(acc_mod)
        
    
    # 5. Noise robustness
    noise_results_orig = {}
    noise_results_mod = {}
    for nl in noise_levels:
        noise_results_orig[nl] = measure_recall_accuracy(hnet, original_patterns, noise_fraction=nl)
        noise_results_mod[nl] = measure_recall_accuracy(hnet, modified_patterns, noise_fraction=nl)
    
    # 6. Confusion matrix
    cm = confusion_matrix_retrieval(hnet, all_patterns)
    

    # 8. Energy trajectorie
    energy_trajectories = {}
    if track_energy_patterns is not None:
        for idx in track_energy_patterns:
            if idx < len(all_patterns):
                init_pattern = all_patterns[idx]
                _, energies = hnet.retrieve_pattern(
                    init_pattern, steps=10, synchronous=True, track_energy=True
                )
                energy_trajectories[idx] = energies
    
    results = {
        "seed": seed,
        "alpha": alpha,
        "flip_probability": flip_probability,
        "original_patterns": original_patterns,
        "modified_patterns": modified_patterns,
        "baseline_acc_original": baseline_acc_original,
        "incremental_acc_original": incremental_acc_original,
        "incremental_acc_modified": incremental_acc_modified,
        "noise_robustness_orig": noise_results_orig,
        "noise_robustness_mod": noise_results_mod,
        "confusion_matrix": cm,
        "energy_trajectories": energy_trajectories
    }
    return results

# ------------------------
# Main Experiment
# ------------------------
def main_experiment_with_replicates(
    n_replicates=5,
    num_neurons=40,
    num_original_patterns=6,
    num_modified_patterns=6,
    alpha=None,
    flip_probability=None,
    num_increments=4,
    noise_levels=None,
    output_folder="results",
    track_energy_patterns=None
):
    os.makedirs(output_folder, exist_ok=True)
    
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    all_results = []
    
    for rep in range(n_replicates):
        seed = 1000 + rep * 50
        result = run_experiment_once(
            seed=seed,
            num_neurons=num_neurons,
            num_original_patterns=num_original_patterns,
            num_modified_patterns=num_modified_patterns,
            alpha=alpha,
            flip_probability=flip_probability,
            num_increments=num_increments,
            noise_levels=noise_levels,
            track_energy_patterns=track_energy_patterns
        )
        all_results.append(result)
        
        # -- 1) Confusion Matrix 
        cm_df = pd.DataFrame(result["confusion_matrix"])
        cm_csv_name = f"confusion_matrix_seed{seed}.csv"
        cm_df.to_csv(os.path.join(output_folder, cm_csv_name), index=False)
        
        plt.figure(figsize=(5.5,4.5))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix (Seed={seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"confusion_matrix_seed{seed}.png"))
        plt.close()
        
        # -- 2) Energy Trajectories 
        if result["energy_trajectories"]:
            plt.figure(figsize=(6,4))
            for idx, energies in result["energy_trajectories"].items():
                plt.plot(energies, marker='o', label=f"Pattern {idx}")
            plt.title(f"Energy Trajectories (Seed={seed})")
            plt.xlabel("Retrieval Step")
            plt.ylabel("Energy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"energy_trajectories_seed{seed}.png"))
            plt.close()

    
    # ------------------------
    # 4) Summaries & Plots Across Replicates
    # ------------------------
    
    # 4a) FlipProb vs Final Modified Accuracy
    overlap_records = []
    for r in all_results:
        final_acc_mod = r["incremental_acc_modified"][-1]  # final step
        mean_mod_acc = np.mean(final_acc_mod)
        overlap_records.append({
            "Seed": r["seed"],
            "FlipProb": r["flip_probability"],
            "Alpha": r["alpha"],
            "MeanModifiedAcc": mean_mod_acc
        })
    df_overlap = pd.DataFrame(overlap_records)
    df_overlap.to_csv(os.path.join(output_folder, "overlap_vs_accuracy.csv"), index=False)
    
    plt.figure(figsize=(5.5,4.5))
    sns.scatterplot(data=df_overlap, x="FlipProb", y="MeanModifiedAcc", 
                    hue="Alpha", palette="viridis", s=80)
    plt.title("FlipProb vs Final Accuracy (Modified)")
    plt.ylim([0,1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "overlap_vs_accuracy_scatter.png"))
    plt.close()
    
    # 4b) Final Increment Accuracy -> Barplot
    # ---------------------------------------
    records = []
    for r in all_results:
        seed = r["seed"]
        alpha_used = r["alpha"]
        flip_used = r["flip_probability"]
        
        final_acc_orig = r["incremental_acc_original"][-1]
        final_acc_mod = r["incremental_acc_modified"][-1]
        
        # Gather per-pattern accuracy
        for i, acc in enumerate(final_acc_orig):
            records.append({
                "Seed": seed,
                "Alpha": alpha_used,
                "FlipProb": flip_used,
                "Type": "Original",
                "PatternIndex": i,
                "Accuracy": acc
            })
        for i, acc in enumerate(final_acc_mod):
            records.append({
                "Seed": seed,
                "Alpha": alpha_used,
                "FlipProb": flip_used,
                "Type": "Modified",
                "PatternIndex": i,
                "Accuracy": acc
            })
    df_final_inc = pd.DataFrame(records)
    df_final_inc.to_csv(os.path.join(output_folder, "final_increment_accuracy.csv"), index=False)
    
    plt.figure(figsize=(6,4))
    sns.barplot(
        data=df_final_inc, 
        x="Seed", 
        y="Accuracy",
        hue="Type", 
        ci='sd', 
        capsize=0.1
    )
    plt.title("Barplot: Final Increment Accuracy")
    plt.ylim([0,1.05])
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "barplot_final_increment_accuracy.png"))
    plt.close()
    
    # 4c) Example of incremental accuracy over increments (line plot for 1 replicate)
    first_rep = all_results[0]
    incr_orig = first_rep["incremental_acc_original"]
    incr_mod = first_rep["incremental_acc_modified"]
    x_vals = np.arange(len(incr_orig)) + 1
    
    plt.figure(figsize=(6,4))
    plt.plot(x_vals, [np.mean(a) for a in incr_orig], marker='o', label='Original')
    plt.plot(x_vals, [np.mean(a) for a in incr_mod], marker='s', label='Modified')
    plt.title(f"Incremental Accuracy (Seed={first_rep['seed']})")
    plt.xlabel("Increment Number")
    plt.ylabel("Mean Accuracy")
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "lineplot_incremental_accuracy_example.png"))
    plt.close()
    
    # 4d) Noise Robustness
    noise_records = []
    for r in all_results:
        for nl in noise_levels:
            orig_acc_list = r["noise_robustness_orig"][nl]
            mod_acc_list = r["noise_robustness_mod"][nl]
            noise_records.append({
                "Seed": r["seed"],
                "Alpha": r["alpha"],
                "FlipProb": r["flip_probability"],
                "NoiseFraction": nl,
                "Type": "Original",
                "MeanAccuracy": np.mean(orig_acc_list)
            })
            noise_records.append({
                "Seed": r["seed"],
                "Alpha": r["alpha"],
                "FlipProb": r["flip_probability"],
                "NoiseFraction": nl,
                "Type": "Modified",
                "MeanAccuracy": np.mean(mod_acc_list)
            })
    df_noise = pd.DataFrame(noise_records)
    df_noise.to_csv(os.path.join(output_folder, "noise_data_extended.csv"), index=False)
    
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df_noise, x="NoiseFraction", y="MeanAccuracy", hue="Type", ci='sd', markers=True)
    plt.title("Noise Robustness Across Replicates")
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "lineplot_noise_robustness_extended.png"))
    plt.close()
    
    return all_results

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    _ = main_experiment_with_replicates(
        n_replicates=3,
        num_neurons=40,
        num_original_patterns=6,
        num_modified_patterns=6,
        alpha=None,
        flip_probability=None,
        num_increments=3,
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4],
        output_folder="results",
        track_energy_patterns=[0, 2]
    )
