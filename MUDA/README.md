# Multi-source Unsupervised Domain Adaptation(MUDA) on PyTorch
This is a PyTorch library for unsupervised domain adaptation with multiple source domains. Here I have implemented some MUDA methods as follows:
* Aligning Domain-specific Distribution and Classifier for Cross-domain

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