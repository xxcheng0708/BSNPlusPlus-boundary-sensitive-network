# BSN++: Complementary Boundary Regressor with Scale-Balanced Relation Modeling for Temporal Action Proposal Generation

A pytorch-version implementation codes of paper:
 "BSN++: Complementary Boundary Regressor with Scale-Balanced Relation Modeling for Temporal Action Proposal Generation",
  which is accepted in AAAI 2021. 

[[Arxiv Preprint]](https://arxiv.org/pdf/2009.07641v4.pdf)

## Result   
| AN     | Recall |
| ------ | ------ |
| AR@1   | 33.7%  |
| AR@5   | 47.8%  |
| AR@10  | 55.0%  |
| AR@100 | 75.3%  |
| AUC    | 66.74   |


![](./img/evaluation_result.png)

## Prerequisites

These code is  implemented in Pytorch 1.5.1 + Python3 . 


## Download Datasets

 The author rescaled the feature length of all videos 
to same length 100, and he provided the rescaled feature at 
 [BaiduCloud](https://pan.baidu.com/s/1vhCPz32rbvdwglC8a1cJeQ) [Code:efy8].


## Training and Testing  of BSN++

All configurations of BSN++ are saved in opts.py, where you can modify training and model parameter.



1. To train the BSN++:
```
python main.py --mode train
```

2. To get the inference proposal of the validation videos and evaluate the proposals with recall and AUC:
```
python main.py --mode inference
```

Of course, you can complete all the process above in one line: 

```
sh bsnpp.sh
```



## Reference

This implementation largely borrows from [BMN](https://github.com/JJBOY/BMN-Boundary-Matching-Network) by [JJBOY](https://github.com/JJBOY).

code:[BMN](https://github.com/JJBOY/BMN-Boundary-Matching-Network)

paper:[BSN++: Complementary Boundary Regressor with Scale-Balanced Relation Modeling for Temporal Action Proposal Generation](https://arxiv.org/pdf/2009.07641v4.pdf)


