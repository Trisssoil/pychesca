import subprocess
import sys
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from pychesca import HAC, SVD
from pychesca.plots import plot_corr, show_dendrogram, plot_svd, heatmap_correlation_cutoffs
from pychesca.visualize import clusters_to_pymol

# Install pychesca
print("Installing pychesca...")
result = subprocess.run([
    sys.executable, "-m", "pip", "install", "git+https://github.com/Dan-Burns/pychesca.git", "-q"
], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ pychesca installed successfully!")
else:
    print("❌ Installation failed:")
    print(result.stderr)

import pychesca
print(f"   Version: {pychesca.__version__}")

# Generate a small synthetic example
rng = np.random.default_rng(42)
n_res = 20
resis = list(range(10, 10 + n_res))
base_a = rng.normal(0, 1, (5, 4))
base_b = rng.normal(0, 1, (5, 4))
noise = rng.normal(0, 0.05, (n_res, 4))
data = np.vstack([np.tile(base_a, (2, 1)), np.tile(base_b, (2, 1))]) + noise
data = np.abs(data) * 0.05

example_df = pd.DataFrame(
    data,
    index=pd.Index(resis, name="RESI"),
    columns=["apo", "state1", "state2", "state3"]
)
example_df.to_csv("chesca_example.csv")
print("✅ Example file saved as 'chesca_example.csv'")
print("   You can download it from the Files panel (📁 icon on the left).")
print()
print("Preview:")
print(example_df.head())

# Parameters
correlation_cutoff = 98.0
linkage_method = "complete"
minimum_cluster_size = 0
output_folder = "chesca_results"

print(f"Parameters set:")
print(f"  Correlation cutoff : {correlation_cutoff}%")
print(f"  Linkage method     : {linkage_method}")
print(f"  Min cluster size   : {minimum_cluster_size if minimum_cluster_size > 0 else '(disabled)'}")
print(f"  Output folder      : {output_folder}/")

# Load CSV (replace with your filename if needed)
chesca_df = pd.read_csv("chesca_example.csv", index_col="RESI")
print(f"\n✅ Loaded 'chesca_example.csv'")
print(f"   Residues: {len(chesca_df)}  |  States: {len(chesca_df.columns)}")
print(f"   States: {list(chesca_df.columns)}")
print()
print("Preview (first 5 rows):")
print(chesca_df.head())

os.makedirs(output_folder, exist_ok=True)

# --- Correlation cutoff explorer ---
print("📊 Generating correlation cutoff heatmap...")
heatmap_correlation_cutoffs(chesca_df, min_corr=90.0, save_file=f"{output_folder}/correlation_cutoff_explorer.pdf")
plt.suptitle("Pairwise Correlation Coefficients (≥ 90%) — use this to choose your cutoff", y=1.01)
plt.tight_layout()
plt.show()
print()

# --- HAC clustering ---
print(f"🔬 Running CHESCA clustering (cutoff = {correlation_cutoff}%, linkage = {linkage_method})...")
min_clust = minimum_cluster_size if minimum_cluster_size > 0 else None
if min_clust is not None:
    hac = HAC(chesca_df, cutoff=correlation_cutoff, method=linkage_method, sub_cluster_cutoff=min_clust)
else:
    hac = HAC(chesca_df, cutoff=correlation_cutoff, method=linkage_method)

print(f"   ✅ Found {hac.n_clusters} clusters across {len(chesca_df)} residues.")
print()

# --- Cluster assignment table ---
cluster_csv_path = f"{output_folder}/cluster_assignments.csv"
hac.clusters.sort_values("cluster").to_csv(cluster_csv_path)
print("📋 Cluster assignments (first 20 rows):")
print(hac.clusters.sort_values("cluster").head(20))
print(f"   Full table saved to: {cluster_csv_path}")
print()

# --- Correlation matrix ---
print("🗺️  Plotting correlation matrix...")
plot_corr(hac, cutoff=correlation_cutoff, save_file=f"{output_folder}/correlation_matrix.pdf")
plt.title(f"CHESCA Correlation Matrix (cutoff = {correlation_cutoff}%)")
plt.show()

# --- Dendrogram ---
print("🌳 Plotting dendrogram...")
show_dendrogram(hac, save_file=f"{output_folder}/dendrogram.pdf")
plt.show()

# --- SVD biplot ---
print("⚡ Running SVD...")
dims = SVD(chesca_df)
plot_svd(dims, centering="column", save_file=f"{output_folder}/svd_plot.pdf")
plt.title("SVD Biplot (column-centered) — circles = residues, diamonds = states")
plt.show()

# --- PyMOL script ---
if min_clust is not None and hasattr(hac, 'sub_cluster_ids'):
    pml_df = hac.clusters[hac.clusters["cluster"].isin(hac.sub_cluster_ids)]
    for cluster_id in hac.sub_cluster_ids:
        sub_resis = hac.clusters[hac.clusters["cluster"] == cluster_id].index
        state_corr = chesca_df.loc[sub_resis].corr().abs()
        hac_states = HAC(state_corr, cluster_states=True)
        show_dendrogram(
            hac_states,
            orientation="top",
            annotate_clusters=False,
            sub_cluster=cluster_id,
            save_file=f"{output_folder}/sub_cluster_{cluster_id}.pdf",
        )
        plt.show()
else:
    pml_df = hac.clusters

pml_path = f"{output_folder}/pymol_selections.pml"
clusters_to_pymol(pml_df, output=pml_path)
print(f"🧪 PyMOL selection script saved to: {pml_path}")
print()
print(f"✅ All done! Results saved to '{output_folder}/'")
