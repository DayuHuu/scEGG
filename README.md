# scEGA: Exogenous Gene Information-Assisted Single-Cell Deep Clustering Framework
![Franework](https://github.com/DayuHuu/scEGA/blob/master/scEGA.png)
**Description:**

scEGA is a deep clustering framework designed for single-cell analysis. It integrates cell and exogenous gene features simultaneously, aligning and fusing them during clustering to generate a more discriminative representation.

**Requirements:**

- Python 3.6.2
- Pandas 1.1.5
- TensorFlow 1.12.0
- Keras 2.1.0
- NumPy 1.19.5
- SciPy 1.5.4
- Scikit-learn 0.19.0

**Datasets:**

- Biase: [GSE57249](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249)
- Darmanis: [PubMed](https://pubmed.ncbi.nlm.nih.gov/26060301/)
- Enge: [PubMed](https://pubmed.ncbi.nlm.nih.gov/28965763/)
- Bjorklund: [GSE70580](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580)
- Sun: [GSE128066](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066)
- Fink: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1534580722004932)

**Examples:**

```python
parser.add_argument('--dataset_str', default='Biase', type=str, help='single cell dataset')
parser.add_argument('--n_clusters', default=3, type=int, help='number of clusters')
parser.add_argument('--label_path', default='data/Biase/label.ann', type=str, help='true labels')

# Add other arguments as needed...


```
**Implement:**
```python
python scEGA.py
```



