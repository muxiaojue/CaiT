# CaiT
This repo is to implement CaiT using mindspore

## Finish:
模型训练中

## To do:
Ascend训练
精度调优
性能调优

## UPDATE:

### 2022.10.27
部分全连接层has_bias设置为False，关闭qkv_bias，训练速度提升明显
在V100GPU上以imagenet-tiny为训练集训练一个epoch耗时1小时

### 2022.10.21
修复cait模型结构，iamgenet验证精度接近
动态图和静态图模式训练跑通

```
{'Top_1_Accuracy': 0.82878, 'Top_5_Accuracy': 0.96392}
```