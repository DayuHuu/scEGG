# scEGG: Exogenous Gene-guided Single-Cell Deep Clustering Framework
![Franework](https://github.com/DayuHuu/scEGA/blob/master/scEGG_framework.png)
**Description:**

scEGG is a deep clustering framework designed for single-cell analysis. It integrates cell and exogenous gene features simultaneously, aligning and fusing them during clustering to generate a more discriminative representation.

**Requirements:**

- Python==3.7.0
- Pandas==1.1.5
- Torch==1.13.1
- NumPy==1.21.6
- SciPy==1.7.3
- Scikit-learn==0.22.2

**Datasets:**

- Darmanis: [PubMed](https://pubmed.ncbi.nlm.nih.gov/26060301/)
- Bjorklund: [GSE70580](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580)
- Sun: [GSE128066](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066)
- Fink: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1534580722004932)

**Examples:**

```python
parser.add_argument('--dataset_str', default='Bjorklund', type=str, help='name of dataset')
parser.add_argument('--n_clusters', default=4, type=int, help='expected number of clusters')

# Add other arguments as needed...


```
**Implement:**
```python
python run_scEGG.py
```



