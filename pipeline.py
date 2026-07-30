import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import EndoDataLoader, adata_to_tensor
from src.models import AttentionClassifier

# --- Output Directories Configuration ---
os.makedirs("output/eda_plots", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)


def run_full_pipeline():
    """
    Executes the end-to-end single-cell genomics pipeline:
    1. Scans and loads all raw dataset files from the data directory.
    2. Performs individual Exploratory Data Analysis (EDA) and saves QC violin plots.
    3. Concatenates all datasets into a unified global atlas using inner join.
    4. Preprocesses the combined atlas (QC, normalization, and highly variable genes selection).
    5. Converts AnnData to PyTorch tensors and initializes the AttentionClassifier model.
    6. Trains the model across multiple epochs and tracks loss convergence.
    7. Extracts top-ranked biomarker genes based on network weights and generates reports/visualizations.
    """
    print("=== Step 1: Scanning and Loading Datasets ===")
    data_path = 'data/'
    loader = EndoDataLoader(data_path=data_path)

    # Retrieve all filtered feature-barcode matrices from the data folder
    all_files = glob.glob(os.path.join(data_path, '*_filtered_feature_bc_matrix.h5'))

    adatas = []
    eda_summary = []

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        print(f"Processing dataset: {file_name}")

        try:
            # Load raw AnnData object
            adata = loader.load_dataset(file_name)
            adata.var_names_make_unique()

            # Capture base statistics for EDA summary reporting
            n_cells_raw = adata.n_obs
            n_genes_raw = adata.n_vars
            label = adata.obs['label'].iloc[0]

            # --- Step 2: Individual Sample EDA & QC Plotting ---
            plt.figure(figsize=(8, 5))
            sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                         multi_panel=True, show=False)
            plt.title(f"QC Metrics: {file_name} (Label: {label})")
            plt.tight_layout()
            plt.savefig(f"output/eda_plots/{file_name}_qc.png", dpi=300)
            plt.close()

            # Preprocess the dataset (QC filtering, normalization, and HVG selection)
            adata = loader.preprocess(adata)

            eda_summary.append({
                'File': file_name,
                'Label': label,
                'Raw_Cells': n_cells_raw,
                'Filtered_Cells': adata.n_obs,
                'Genes_Retained': adata.n_vars
            })

            adatas.append(adata)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if not adatas:
        raise ValueError("No datasets successfully loaded!")

    # Save comprehensive EDA summary report to CSV
    eda_df = pd.DataFrame(eda_summary)
    eda_df.to_csv("output/reports/eda_summary_report.csv", index=False)
    print("\n[✔] EDA Summary Report saved to output/reports/eda_summary_report.csv")

    print("\n=== Step 3: Global Atlas Concatenation ===")
    # Concatenate all datasets using inner join to align common gene features
    combined_adata = sc.concat(adatas, axis=0, join='inner')
    combined_adata.obs_names_make_unique()

    print(f"Combined Atlas Shape -> Cells: {combined_adata.n_obs}, Genes: {combined_adata.n_vars}")
    print("Label distribution across all data:")
    print(combined_adata.obs['label'].value_counts())

    # --- Step 4: Tensor Conversion ---
    print("\n=== Step 4: Preparing Tensors for Model Training ===")
    X_tensor = adata_to_tensor(combined_adata)
    y_tensor = torch.tensor(combined_adata.obs['label'].values, dtype=torch.long)

    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")

    # --- Step 5: Model Initialization and Training ---
    print("\n=== Step 5: Training AttentionClassifier on the Global Atlas ===")
    input_dim = X_tensor.shape[1]
    model = AttentionClassifier(input_dim=input_dim, hidden_dim=128, num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs, weights = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save training loss convergence curve plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='b')
    plt.title("Model Training Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.savefig("output/reports/training_loss_curve.png", dpi=300)
    plt.close()

    # --- Step 6: Biomarker Discovery & Biological Insights ---
    print("\n=== Step 6: Extracting Top Biomarkers and Generating Reports ===")
    model.eval()
    with torch.no_grad():
        # Extract weights from the first linear layer connecting the input genes
        first_layer_weights = model.feature_extractor[0].weight.abs()
        gene_importance = first_layer_weights.mean(dim=0).detach().cpu().numpy()

    gene_importance_df = pd.DataFrame({
        'Gene': combined_adata.var_names,
        'Importance_Score': gene_importance
    }).sort_values(by='Importance_Score', ascending=False)

    # Save full ranked biomarker list to CSV
    gene_importance_df.to_csv("output/reports/top_biomarkers_ranked.csv", index=False)

    # Save bar plot of the top 20 biomarker genes
    plt.figure(figsize=(10, 6))
    top_20 = gene_importance_df.head(20)
    sns.barplot(x='Importance_Score', y='Gene', data=top_20, palette='viridis')
    plt.title("Top 20 Biomarker Genes Identified by Attention Model")
    plt.xlabel("Importance Score")
    plt.ylabel("Gene Symbol")
    plt.tight_layout()
    plt.savefig("output/reports/top_20_biomarkers_barplot.png", dpi=300)
    plt.close()

    print("\n[✔] Pipeline executed successfully!")
    print("[✔] All EDA plots, loss curves, and biomarker ranking reports saved under the 'output/' directory.")


if __name__ == '__main__':
    run_full_pipeline()