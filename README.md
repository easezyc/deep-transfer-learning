# Deep Transfer Learning on PyTorch
This is a PyTorch library for deep transfer learning. And this is a part of another repository---[transferlearning](https://github.com/jindongwang/transferlearning) which I will merge this into. If you need more code, please refer to [code](https://github.com/jindongwang/transferlearning/tree/master/code/deep). Here I have implemented some deep transfer methods as follows:
* DAN: Learning Transferable Features with Deep Adaptation Networks
* Deep Coral: Deep CORAL Correlation Alignment for Deep Domain Adaptation

## Results on Office31
| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:----:|:----:|:-------:|
| DAN | 83.8±0.4 | 96.8±0.2 | 99.5±0.1 | 78.4±0.2 | 66.7±0.3 | 62.7±0.2 | 81.3 |
| DCORAL | 77.7±0.3 | 97.6±0.2 | 99.7±0.1 | 81.1±0.4 | 64.6±0.3 | 64.0±0.4 | 80.8 |

## Contact
If you have any problem about this library, please create an Issue or send us an Email at:
* zhuyc0204@gmail.com