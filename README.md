DCD: A Deep Learning-Based Community Detection Software for Large-scale Networks
=========================================================

DCD (Deep learning-based Community Detection) is designed to apply state-of-the-art deep learning technologies to identify communities for large-scale networks. Compared with existing community detection methods, DCD offers a unified solution for many variations of community detection problems.  

DCD provides 4 implementation of community detection, 1 evaluation, and two types of networked data:


| Function      | Description       | Input | Output |
|------------|-------------------------------|-----------|---------|
| KMeans     | Clustering baseline method (1) | Network node file <br/> Network edge file <br/> K | <node id, community id> |
| MM      | Clustering baseline method (2) | Network node file <br/> Network edge file | <node id, community id> |
| DCD     | DCD | Network node file <br/> Network edge file <br/> K | <node id, community id> |
| DCD+    | Variant of GCN with node attributes | Network node file with attributes <br/> Network edge file <br/> K | <node id, community id> |
| Evaluation | Evaluate the performance | Network node file <br/> Network edge file <br/> Community assignment | performance value|
| Random network | Generate random network datasets | Network size <br/> Community size <br/> Probability of edges within communities <br/> Probability of edges between communities <br/> Directed network flag | <node id, community id> <br/> Network node file <br/> Network edge file |
| Facebook network  | Import Facebook brand-brand network  | None| Facebook dataset |
| Citation network  | Import citation network  | None| [Citation] dataset |

[Citation]: https://snap.stanford.edu/data/cit-HepTh.html

Performance
------------

Performance comparison on four random networks. Note: numbers in parentheses are running time (seconds).

| Network size    | Community size  | K-Means | Modularity <br/> Maximization| DCD   |
|-----------------|-----------------|---------|------------------------|---------------|
| 100       | 10  | 0.561<br/>(0.07) | 0.922<br/>(0.01) |0.826<br/>(0.01)|
| 1,000     | 100 | 0.699<br/>(1.04) | 0.807<br/>(1.11) |0.935<br/>(0.11)|
| 10,000    | 100 | 0.726<br/>(199.90) | 0.633<br/>(338.82) |0.845<br/>(62.30)|
| 20,000    | 100 | 0.709<br/>(807.56) | 0.702<br/>(1666.59) |0.814<br/>(444.12)|


Performance comparison on two real-world networks. Note: numbers in parentheses are running time (seconds).

| Network   | Community size  | K-Means | Modularity <br/> Maximization| DCD  | DCD+|
|-----------------|-----------------|---------|------------------------|------|-----|
|       | 50  | 0.451<br/>(82.72) |  /   |0.503<br/>(38.46)| 0.532<br/>(39.01) |
|       | 100 | 0.427<br/>(103.91) |   /  |0.519<br/>(37.86)| 0.520<br/>(38.95) |
| Facebook <br/> weighted and undirected <br/>network with node attributes | 150 | 0.406<br/>(118.58) |   /  |0.532<br/>(37.87)| 0.525<br/>(38.92) |
|       | 200 | 0.383<br/>(144.77) |  /   |0.521<br/>(37.87)| 0.530<br/>(39.50)|
|       | 33(mm)| 0.464<br/>(75.47) |  0.516<br/>(64.70) |0.521<br/>(38.05)| 0.538<br/>(39.20) |
|       |       |   |   |   |   |
|       | 100 | 0.438<br/>(446.27) |  /   |0.897<br/>(216.36)||
|       | 200 | 0.447<br/>(596.94) |   /  |0.916<br/>(216.37)||
| Facebook <br/> weighted and undirected <br/>network with node attributes | 500 | 0.561<br/>(1096.14) |   /  |0.927<br/>(216.59)| No node attributes |
|       | 1,000 | 0.611<br/>(1843.03) |  /   |0.940<br/>(217.32)| |
|       | 2,078 (mm)  | 0.660<br/>(3219.14) |  0.790<br/>(715.26) |0.939<br/>(217.56)| |


Requirements
------------

Generally, the library is compatible with Python 3.6/3.7.


Installation
------------

### From Conda ###

```bash
conda install -c dcd
```

### From Source ###

Before installation, make sure you have `conda` installed.

```bash
git clone https://github.com/kpzhang/deepcommunitydetection
cd deepcommunitydetection
conda install -y --file conda/requirements.txt
mkdir build
cd build && cmake .. && make && cd -
cd python && python setup.py install && cd -
```

Quick Start
-----------

Here is a quick-start example.

```bash
graphvite baseline quick start
```

Typically, the example takes no more than 1 minute. You will obtain some output like

```
Batch id: 6000
loss = 0.371041

------------- link prediction --------------
AUC: 0.899933

----------- node classification ------------
macro-F1@20%: 0.242114
micro-F1@20%: 0.391342
```

Baseline Benchmark
------------------

To reproduce a baseline benchmark, you only need to specify the keywords of the
experiment. e.g. model and dataset.

```bash
graphvite baseline [keyword ...] [--no-eval] [--gpu n] [--cpu m] [--epoch e]
```

You may also set the number of GPUs and the number of CPUs per GPU.

Use ``graphvite list`` to get a list of available baselines.


Development Team
----------------

GraphVite is developed by Prof.[Kunpeng Zhang], Prof. [Shaokun Fan], and Prof. [Bruce Golden].

[Kunpeng Zhang]: http://www.terpconnect.umd.edu/~kpzhang/
[Shaokun Fan]: https://business.oregonstate.edu/users/shaokun-fan
[Bruce Golden]: http://scholar.rhsmith.umd.edu/bgolden/home

Citation
--------

If you find this useful for your research or development, please cite our work.
