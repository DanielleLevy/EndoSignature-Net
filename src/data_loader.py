import scanpy as sc
import os
import torch

class EndoDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_dataset(self, file_name: str):
        """
        Loads dataset. Handles .h5ad files or 10x Genomics .h5 files.
        """
        full_path = os.path.join(self.data_path, file_name)

        if file_name.endswith('.h5ad'):
            adata = sc.read_h5ad(full_path)
        elif file_name.endswith('.h5'):
            adata = sc.read_10x_h5(full_path)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")

        # Determine label based on filename
        if 'Ctrl' in file_name:
            adata.obs['label'] = 0
        elif 'E' in file_name:
            adata.obs['label'] = 1
        else:
            adata.obs['label'] = -1

        print(
            f"Successfully loaded: {file_name} | Cells: {adata.n_obs} | Genes: {adata.n_vars} | Label: {adata.obs['label'].iloc[0]}")
        return adata

    def preprocess(self, adata):
        """
        Performs Quality Control, Normalization, and Highly Variable Gene selection.

        Args:
            adata (AnnData): The raw dataset.

        Returns:
            adata (AnnData): The preprocessed dataset with variable genes selected.
        """
        # 1. Quality Control: Identify mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

        # 2. Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # 3. Normalization: Normalize total counts to 10,000 per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # 4. Feature Selection: Keep the 2,000 most variable genes
        # This reduces noise and prepares the data for the Attention model
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

        print(f"Preprocessing complete. Remaining genes: {adata.n_vars}")
        return adata

def adata_to_tensor(adata):
        """
        Converts AnnData object to PyTorch Tensor.

        Args:
            adata: The preprocessed AnnData object.

        Returns:
            torch.Tensor: The expression matrix as a tensor.
        """
        # .X contains the expression matrix in AnnData
        data = adata.X

        # If the data is sparse (common in single-cell), convert to dense
        if hasattr(data, 'toarray'):
            data = data.toarray()

        return torch.tensor(data, dtype=torch.float32)