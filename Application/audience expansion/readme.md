# Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising
This is an official implementation for **Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising** which has been accepted by KDD2021.

## Introduction

In recommender systems and advertising platforms, marketers always want to deliver products, contents, or advertisements to potential audiences over media channels such as display, video, or social. Given a set of audiences or customers (seed users), the audience expansion technique (look-alike modeling) is a promising solution to identify more potential audiences, who are similar to the seed users and likely to finish the business goal of the target campaign. However, look-alike modeling faces two challenges: (1) In practice, a company could run hundreds of marketing campaigns to promote various contents within completely different categories every day, e.g., sports, politics, society. Thus, it is difficult to utilize a common method to expand audiences for all campaigns. (2) The seed set of a certain campaign could only cover limited users. Therefore, a customized approach based on such a seed set is likely to be overfitting.
  
In this paper, to address these challenges, we propose a novel two-stage framework named Meta Hybrid Experts and Critics (MetaHeac) which has been deployed in WeChat Look-alike System. In the offline stage, a general model which can capture the relationships among various tasks is trained from a meta-learning perspective on all existing campaign tasks. In the online stage, for a new campaign, a customized model is learned with the given seed set based on the general model. According to both offline and online experiments, the proposed MetaHeac shows superior effectiveness for both content marketing campaigns in recommender systems and advertising campaigns in advertising platforms. Besides, MetaHeac has been successfully deployed in WeChat for the promotion of both contents and advertisements, leading to great improvement in the quality of marketing.

## Requirements

- Python 3.6
- Pytorch > 1.0
- Pandas
- Numpy

## File Structure

```
.
├── code
│   ├── main.py             # Entry function
│   ├── model.py            # Models
│   ├── metamodel.py        # Training Model from a meta-learning perspective
│   ├── readme.md
│   └── run.py              # Training and Evaluating 
│   └── utils.py            # Some auxiliary classes
└── data
    ├── process.py          # Preprocess the original data
    ├── processed_data      # The folder to contain the processed data
```

## Dataset

We utilized the Tencent Look-alike Dataset. 
To download the dataset, you can use the following link: [Tencent Look-alike Dataset](https://algo.qq.com/archive.html?). Then put the data in `./data`.

You can use the following command to preprocess the dataset. 
The final data will be under `./data/processed_data`.

```python
python process.py
```

## Run

Parameter Configuration:

- task_count: the number of tasks in a mini-batch, default for `5`
- num_expert: the number of experts, default for `8`
- num_output: the number of critics, default for `5`
- seed: random seed, default for `2020`
- gpu: the index of gpu you will use, default for `0`
- batchsize: default for `512`

You can run this model through:

```powershell
python main.py --task_count 5 --num_expert 8 --output 5 --batchsize 512
```

## Reference

```
Zhu Y, Liu Y, Xie R, et al. Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising[C]. KDD, 2021.
```

or in bibtex style:

```
@inproceedings{zhu2021learning,
  title={Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising},
  author={Zhu, Yongchun and Liu, Yudan and Xie, Ruobing and Zhuang, Fuzhen and Hao, Xiaobo and Ge, Kaikai and Zhang, Xu and Lin, Leyu and Cao, Juan},
  booktitle={KDD},
  year={2021}
}
```