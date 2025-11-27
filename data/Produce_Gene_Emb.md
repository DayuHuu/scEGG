# Data Preprocessing and Network Embedding Tutorial

This tutorial outlines the workflow for processing scRNA-seq data (using the Bjorklund dataset), extracting biological networks via the STRING database, and preparing the data for Node2vec embedding generation.

## Prerequisites

Ensure you have the following Python libraries installed:

```bash
pip install scanpy pandas openpyxl scipy
```

**Directory Structure:**

```text
project_root/
│
├── Bjorklund/
│   ├── data.tsv   (Original expression matrix)
│   └── ...
└── script.py
```

-----

## Step 1: Identify Highly Variable Genes

In this step, we normalize the raw data and select the most informative genes (highly variable genes). These genes are saved to an Excel file for network extraction.

```python
import scanpy as sc
import pandas as pd
import os

# Define dataset and paths
dataset = 'Bjorklund'
data_dir = './' + dataset
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

data_path = os.path.join(data_dir, 'data.tsv')

# Load data
# Assuming data.tsv has genes as rows and cells as columns; transpose (.T) if necessary
adata = sc.read_csv(data_path, delimiter='\t', first_column_names=True).T

# Normalize data
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes (Top 500)
sc.pp.highly_variable_genes(adata, n_top_genes=500)

# Get the names of the highly variable genes
highly_variable_genes = adata.var_names[adata.var['highly_variable']]

# Convert gene names to uppercase for consistency
highly_variable_genes_upper = highly_variable_genes.str.upper()

# Save the highly variable genes to an Excel file
highly_variable_genes_df = pd.DataFrame(highly_variable_genes_upper, columns=['Gene Name'])
highly_variable_genes_df.to_excel(os.path.join(data_dir, 'gene.xlsx'), header=None, index=False)

print("Step 1 Complete: Highly variable genes saved to gene.xlsx in uppercase.")
```

-----

## Step 2: Extract Network Information via STRING

This step involves manual interaction with the STRING database to retrieve Protein-Protein Interaction (PPI) networks for the genes identified in Step 1.

1.  Navigate to the **STRING Database**: [https://cn.string-db.org/](https://cn.string-db.org/)
2.  Select **"Multiple Proteins"** from the left-hand menu.
3.  Click **"Choose File"** and upload the `Bjorklund/gene.xlsx` file generated in Step 1.
4.  In the **Organism** field, select the appropriate species (e.g., *Homo sapiens*).
5.  Click **"Search"** and proceed until the network visualization appears.
6.  Click on the **"Exports"** tab.
7.  Download the network interaction data (e.g., as a TSV file).
8.  **Post-processing**: Convert the downloaded file into an Excel file named `Bjorklund_net.xlsx`.
      * Ensure the file contains columns representing the interacting nodes (e.g., `node1` and `node2`).
      * Save this file inside the `Bjorklund/` directory.

> **Note:** Some genes may not have matching entries in the STRING database; this is expected behavior.

-----

## Step 3: Network Filtering and Node Alignment

We must now filter the network to ensure that all nodes in the network exist in our expression data and vice-versa. We will also generate the edge list required for Node2vec.

```python
import pandas as pd
import os

dataset = 'Bjorklund'
data_dir = './' + dataset

# 1. Read the original data.tsv file
# Assuming index_col=0 is the gene name
data = pd.read_csv(os.path.join(data_dir, 'data.tsv'), delimiter='\t', index_col=0)
data.index = data.index.str.upper()

# 2. Read the gene list (High Variable Genes)
gene_list = pd.read_excel(os.path.join(data_dir, 'gene.xlsx'), header=None)[0]

# 3. Filter the expression data using the gene list
filtered_data = data.loc[data.index.isin(gene_list)]

# 4. Read the network data obtained from STRING
network_data = pd.read_excel(os.path.join(data_dir, f'{dataset}_net.xlsx'))

# 5. Filter the network data
# Keep edges only if both node1 and node2 are present in our filtered expression data
filtered_network = network_data[
    network_data['node1'].str.upper().isin(filtered_data.index) &
    network_data['node2'].str.upper().isin(filtered_data.index)
]

# 6. Create the final gene list (Intersection of Network and Data)
genelist_f = pd.concat([
    filtered_network['node1'].str.upper(),
    filtered_network['node2'].str.upper()
]).unique()

# 7. Create the final expression dataset
final_data = filtered_data.loc[filtered_data.index.isin(genelist_f)]

# Aggregate duplicate genes by calculating the mean
final_data_aggregated = final_data.groupby(final_data.index).mean()

# Save the aggregated data to CSV
final_data_aggregated.to_csv(os.path.join(data_dir, f'{dataset}_data.csv'))
print(f"Processed expression data saved. Total genes: {final_data_aggregated.shape[0]}")

# 8. Convert gene names to numeric IDs for Node2vec
unique_genes = pd.Series(list(filtered_data.index)).sort_values().reset_index(drop=True)
gene_to_id = {gene: i for i, gene in enumerate(unique_genes)}

# Apply mapping to network nodes
filtered_network['node1_id'] = filtered_network['node1'].str.upper().map(gene_to_id)
filtered_network['node2_id'] = filtered_network['node2'].str.upper().map(gene_to_id)

# Select only the ID columns
network_id_only = filtered_network[['node1_id', 'node2_id']]

# Save to .txt file (Tab-separated, No Header, No Index)
output_edge_path = os.path.join(data_dir, f'{dataset}_node2vec.txt')
network_id_only.to_csv(output_edge_path, sep='\t', index=False, header=False)

# Count unique nodes in the network
unique_nodes_count = pd.concat([network_id_only['node1_id'], network_id_only['node2_id']]).nunique()
print("Number of unique nodes in the network:", unique_nodes_count)
print(f"Edge list saved to {output_edge_path}")
```

-----

## Step 4: Generate Embeddings with Node2vec

Finally, use the generated edge list to create gene embeddings.

**Reference:**
Please refer to the official Node2vec implementation:

  * [Node2vec GitHub Repository](https://github.com/aditya-grover/node2vec)

**Usage Example:**

Assuming you have cloned the repository, you can run the following command in your terminal:

```bash
python src/main.py --input ./Bjorklund/Bjorklund_node2vec.txt --output ./Bjorklund/Bjorklund.emb --weighted --directed
```

**Parameters:**

  * `--input`: Path to the edge list generated in Step 3.
  * `--output`: Path where the embeddings will be saved.
  * Additional parameters (e.g., `--dimensions`, `--walk-length`, `--num-walks`) can be adjusted based on your specific requirements.

