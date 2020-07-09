# Deep Transfer Learning on PyTorch
This is a PyTorch library for deep transfer learning. We divide the code into two
aspects: Single-source Unsupervised Domain Adaptation (SUDA) and Multi-source Unsupervised Domain Adaptation (MUDA). There are many SUDA methods, however I find there is a few MUDA methods with deep learning. Besides, MUDA with deep learning might be a more promising direction for domain adaptation.

Here I have implemented some deep transfer methods as follows:
* UDA
    * DDC：Deep Domain Confusion Maximizing for Domain Invariance
    * DAN: Learning Transferable Features with Deep Adaptation Networks (ICML2015)
    * Deep Coral: Deep CORAL Correlation Alignment for Deep Domain Adaptation (ECCV2016)
    * Revgrad: Unsupervised Domain Adaptation by Backpropagation (ICML2015)
    * MRAN: Multi-representation adaptation network for cross-domain image classification (Neural Network 2019)
    * DSAN: Deep Subdomain Adaptation Network for Image Classification (IEEE Transactions on Neural Networks and Learning Systems 2020)
* MUDA
    * Aligning Domain-specific Distribution and Classifier for Cross-domain Classification from Multiple Sources (AAAI2019)
* Application
    * Cross-domain Fraud Detection: Modeling Users’ Behavior Sequences with Hierarchical Explainable Network for Cross-domain Fraud Detection (WWW2020)
* Survey
    * [A Comprehensive Survey on Transfer Learning](https://arxiv.org/abs/1911.02685) (Proc. IEEE)


## Results on Office31(UDA)
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| ResNet | 68.4±0.5 | 96.7±0.5 | 99.3±0.1 | 68.9±0.2 | 62.5±0.3 | 60.7±0.3 | 76.1 |
| DDC | 75.8±0.2 | 95.0±0.2 | 98.2±0.1 | 77.5±0.3 | 67.4±0.4 | 64.0±0.5 | 79.7 |
| DDC\* | 78.3±0.4 | 97.1±0.1 | 100.0±0.0 | 81.7±0.9 | 65.2±0.6 | 65.1±0.4 | 81.2 |
| DAN | 83.8±0.4 | 96.8±0.2 | 99.5±0.1 | 78.4±0.2 | 66.7±0.3 | 62.7±0.2 | 81.3 |
| DAN\* | 82.6±0.7 | 97.7±0.1 | 100.0±0.0 | 83.1±0.9 | 66.8±0.3 | 66.6±0.4 | 82.8 |
| DCORAL\* | 79.0±0.5 | 98.0±0.2 | 100.0±0.0 | 82.7±0.1 | 65.3±0.3 | 64.5±0.3 | 81.6 |
| Revgrad | 82.0±0.4 | 96.9±0.2 | 99.1±0.1 | 79.7±0.4 | 68.2±0.4 | 67.4±0.5 | 82.2 |
| Revgrad\* | 82.6±0.9 | 97.8±0.2 | 100.0±0.0 | 83.3±0.9 | 66.8±0.1 | 66.1±0.5 | 82.8 |
| MRAN | 91.4±0.1 | 96.9±0.3 | 99.8±0.2 | 86.4±0.6 | 68.3±0.5 | 70.9±0.6 | 85.6 |
| DSAN | 93.6±0.2 | 98.4±0.1 | 100.0±0.0 | 90.2±0.7 | 73.5±0.5 | 74.8±0.4 | 88.4 |

> Note that the results without '\*' comes from [paper](http://ise.thss.tsinghua.edu.cn/~mlong/doc/multi-adversarial-domain-adaptation-aaai18.pdf). The results with '\*' are run by myself with the code. 

## Results on Office31(MUDA)
| Standards | Method | A,W - D | A,D - W | D,W - A | Average |
|:--------------:|:--------------:|:-----:|:-----:|:-----:|:-------:|
| | ResNet | 99.3 | 96.7 | 62.5 | 86.2 |
|  | DAN | 99.5 | 96.8 | 66.7 | 87.7 |
| Single Best| DCORAL | 99.7 | 98.0 | 65.3 | 87.7 |
|  | RevGrad | 99.1 | 96.9 | 68.2 | 88.1 |
||
|  | DAN | 99.6 | 97.8 | 67.6 | 88.3 |
| Source Combine | DCORAL | 99.3 | 98.0 | 67.1 | 88.1 |
|  | RevGrad | 99.7 | 98.1 | 67.6 | 88.5 |
||
| Multi-Source | MFSAN | 99.5 | 98.5 | 72.7 | 90.2 |

## Results on OfficeHome(MUDA)
| Standards | Method | C,P,R - A | A,P,R - C | A,C,R - P | A,C,P - R | Average |
|:--------------:|:--------------:|:-----:|:-----:|:-----:|:-----:|:-------:|
| | ResNet | 65.3 | 49.6 | 79.7 | 75.4 | 67.5 |
|  | DAN | 64.1 | 50.8 | 78.2 | 75.0 | 67.0 |
| Single Best | DCORAL | 68.2 | 56.5 | 80.3 | 75.9 | 70.2 |
|  | RevGrad | 67.9 | 55.9 | 80.4 | 75.8 | 70.0 |
||
|  | DAN | 68.5 | 59.4 | 79.0 | 82.5 | 72.4 |
| Source Combine | DCORAL | 68.1 | 58.6 | 79.5 | 82.7 | 72.2 |
|  | RevGrad | 68.4 | 59.1 | 79.5 | 82.7 | 72.4 |
||
| Multi-Source | MFSAN | 72.1 | 62.0 | 80.3 | 81.8 | 74.1 |

> Note that  (1) Source combine: all source domains are combined together into a traditional single-source v.s. target setting. (2) Single best: among the multiple source domains, we report the best single source transfer results. (3) Multi-source: the results of MUDA methods.

## Note
> If you find that your accuracy is 100%, the problem might be the dataset folder. Please note that the folder structure required for the data provider to work is:
```
-dataset
    -amazon
    -webcam
    -dslr
```


## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
* zhuyongchun18s@ict.ac.cn
* jindongwang@outlook.com