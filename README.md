# DartsReNet: Exploring new RNN cells in ReNet architectures
**Authors:** [Brian B. Moser](https://www.dfki.de/en/web/about-us/employee/person/brmo01/), [Federico Raue](https://www.dfki.de/en/web/about-us/employee/person/fera02/), [Jörn Hees](https://www.dfki.de/en/web/about-us/employee/person/johe02/), [Andreas Dengel](https://www.dfki.de/en/web/about-us/employee/person/ande00/)

**Institutes:** [German Research Center for Artificial Intelligence (DFKI)](https://www.dfki.de/en/web/) and [TU Kaiserslautern](https://www.uni-kl.de/en/), Germany

The code was produced as part of the paper "DartsReNet: Exploring new RNN cells in ReNet architectures", published at [ICANN2020](https://e-nns.org/icann2020/).
The goal of this work was to derive novel RNN cells for Image Classification. In contrast to standard RNN cell designs like LSTM and GRU, we found cells that are beneficial for two dimensional sequences (images). We accomplished this by using DARTS, a fast NAS approach, and ReNet, an RNN alternative to convolution layers.
This repository was used to derive and to evaluate new cells.

## Background
The basis is given by the two papers:

- [ReNet](https://arxiv.org/abs/1505.00393)
- [DARTS](https://arxiv.org/abs/1806.09055)

We implemented ReNet ourself in [PyTorch](https://pytorch.org/) and modified the code of DARTS (GitHub-Link: https://github.com/quark0/darts).


## Requirements
We have the same constraints as DARTS since we were extending the code of the authors.
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```
**NOTE:** PyTorch 0.4+ is not supported at this moment and would lead to OOM.

## Finding new Cells
We used [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) for Cell Search. For Cell Evaluation, we used CIFAR-10 and [SVHN](http://ufldl.stanford.edu/housenumbers/).

### Cell Search
Finding new cells can be done by using a single command:
```
python train_search.py --rnn_cell="vanilla"
```

The RNN cell types denotes the different variants we implemented. For further details, please take a look in our paper. Others are:
```
python train_search.py --rnn_cell="sigmoid"
python train_search.py --rnn_cell="directional"
```
The new cell designs are copy pasted to the file
```
genotypes.py
```

### Cell Evaluation
The cell derived have to saved in the genotypes.py file. We provied the cell designs of the paper (DARTS_Vanilla, DARTS_Sigmoid and DARTS_Directional).
Evaluation on CIFAR-10 is done with this command:
```
python train.py --rnn_cell="vanilla" --arch=DARTS_Vanilla
```
RNN Cell refers again to the cell variant and arch refers to the cell design derived by the Cell Search (see Section above).
Again, there are also the other two variants:
```
python train.py --rnn_cell="sigmoid" --arch=DARTS_Sigmoid
python train.py --rnn_cell="directional" --arch=DARTS_Directional
```

## Citation
Please cite our [paper](https://link.springer.com/chapter/10.1007/978-3-030-61609-0_67), if you use it:
```
@inproceedings{moser2020dartsrenet,
  title={DartsReNet: Exploring New RNN Cells in ReNet Architectures},
  author={Moser, Brian B and Raue, Federico and Hees, J{\"o}rn and Dengel, Andreas},
  booktitle={International Conference on Artificial Neural Networks},
  pages={850--861},
  year={2020},
  organization={Springer}
}
```
