DCD: A Deep Learning-Based Community Detection Software for Large-scale Networks
=========================================================

DCD (Deep learning-based Community Detection) is designed to apply state-of-the-art deep learning technologies to identify communities for large-scale networks. Compared with existing community detection methods, DCD offers a unified solution for many variations of community detection problems.  

DCD provides 4 implementation of community detection, 1 evaluation, and two types of networked data:


| Function      | Description       | Input | Output |
|------------|-------------------------------|-----------|---------|
| KMeans     | Clustering baseline method (1) | Network node file <br/> Network edge file <br/> K | <node id, community id> |
| MM      | Clustering baseline method (2) | Network node file <br/> Network edge file | <node id, community id> |
| GCN     | DCD | Network node file <br/> Network edge file <br/> K | <node id, community id> |
| GCN+    | Variant of GCN with node attributes | Network node file with attributes <br/> Network edge file <br/> K | <node id, community id> |
| Evaluation | Evaluate the performance | Network node file <br/> Network edge file <br/> Community assignment | performance value|
| Random network | Generate random network datasets | Network size <br/> Community size <br/> Probability of edges within communities <br/> Probability of edges between communities <br/> Directed network flag | <node id, community id> <br/> Network node file <br/> Network edge file |
| Facebook network  | Import Facebook brand-brand network  | None| Facebook dataset |
| Citation network  | Import citation network  | None| [Citation] dataset |

[Citation]: https://snap.stanford.edu/data/cit-HepTh.html

Performance comparison on four random networks. Note: numbers in parentheses are running time (seconds).

| Network size    | Community size  | Baselines                        | Our method    |
|-----------------|-----------------|----------------------------------|---------------|
|                 |                 |K-Means | modularity maximization | GCN           |      
| 100       | 10 | 13.5 mins / 54.3 s | 5.82x / 1.93x |
| 1,000       | 100 | 28.1 mins / 55.8 s | 7.88x / 4.50x |
| 10,000       | 100 | 28.1 mins / 55.8 s | 7.88x / 4.50x |
| 20,000       | 100 | 28.1 mins / 55.8 s | 7.88x / 4.50x |


Training time of high-dimensional data visualization on [MNIST] dataset.

| Model        | Existing Implementation       | GraphVite | Speedup |
|--------------|-------------------------------|-----------|---------|
| [LargeVis]   | [15.3 mins (CPU parallel)][5] | 13.9 s    | 66.8x   |

[MNIST]: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
[LargeVis]: https://arxiv.org/pdf/1602.00370.pdf
[5]: https://github.com/lferry007/LargeVis

Requirements
------------

Generally, GraphVite works on any Linux distribution with CUDA >= 9.2.

The library is compatible with Python 2.7 and 3.6/3.7.

Installation
------------

### From Conda ###

```bash
conda install -c milagraph graphvite cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")
```

If you only need embedding training without evaluation, you can use the following
alternative with minimal dependencies.

```bash
conda install -c milagraph graphvite-mini cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")
```

### From Source ###

Before installation, make sure you have `conda` installed.

```bash
git clone https://github.com/DeepGraphLearning/graphvite
cd graphvite
conda install -y --file conda/requirements.txt
mkdir build
cd build && cmake .. && make && cd -
cd python && python setup.py install && cd -
```

### On Colab ###

```bash
!wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local -f

!conda install -y -c milagraph -c conda-forge graphvite \
    python=3.6 cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+\.\d+")
!conda install -y wurlitzer ipykernel
```

```python
import site
site.addsitedir("/usr/local/lib/python3.6/site-packages")
%reload_ext wurlitzer
```

Quick Start
-----------

Here is a quick-start example of the node embedding application.

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

Custom Experiment
-----------------

Create a yaml configuration scaffold for graph, knowledge graph, visualization or
word graph.

```bash
graphvite new [application ...] [--file f]
```

Fill some necessary entries in the configuration following the instructions. You
can run the configuration by

```bash
graphvite run [config] [--no-eval] [--gpu n] [--cpu m] [--epoch e]
```

High-dimensional Data Visualization
-----------------------------------

You can visualize your high-dimensional vectors with a simple command line in
GraphVite.

```bash
graphvite visualize [file] [--label label_file] [--save save_file] [--perplexity n] [--3d]
```

The file can be either a numpy dump `*.npy` or a text matrix `*.txt`. For the save
file, we recommend to use `png` format, while `pdf` is also supported.

Contributing
------------

We welcome all contributions from bug fixs to new features. Please let us know if you
have any suggestion to our library.

Development Team
----------------

GraphVite is developed by Prof. Kunpeng Zhang, Prof. Shaokun Fan, and Prof. Bruce Golden.

Authors of this project are [Zhaocheng Zhu], [Shizhen Xu], [Meng Qu] and [Jian Tang].
Contributors include [Kunpeng Wang] and [Zhijian Duan].

[MilaGraph]: https://github.com/DeepGraphLearning
[Zhaocheng Zhu]: https://kiddozhu.github.io
[Shizhen Xu]: https://github.com/xsz
[Meng Qu]: https://mnqu.github.io
[Jian Tang]: https://jian-tang.com
[Kunpeng Wang]: https://github.com/Kwinpeng
[Zhijian Duan]: https://github.com/zjduan

Citation
--------

If you find this useful for your research or development, please cite our work.
