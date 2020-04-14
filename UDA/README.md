# Unsupervised Domain Adaptation(UDA) on PyTorch
This is a PyTorch library for unsupervised domain adaptation with single source domain. Here I have implemented some unsupervised domain adaptation methods as follows:
* DDC：Deep Domain Confusion Maximizing for Domain Invariance
* DAN: Learning Transferable Features with Deep Adaptation Networks
* Deep Coral: Deep CORAL Correlation Alignment for Deep Domain Adaptation
* Revgrad: Unsupervised Domain Adaptation by Backpropagation
* MRAN: Multi-representation adaptation network for cross-domain image classification
* Deep Subdomain Adaptation Network for Image Classification

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