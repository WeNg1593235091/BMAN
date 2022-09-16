# Bidirectional Matching and Aggregation Network


## Dependencies

The code is written in Python 3.8 and pytorch 11.0.0.


## Evaluation Results

Model | 5 Way 1 Shot | 5 Way 5 Shot | 10 Way 1 Shot | 10 Way 5 Shot
----- | ------------ | ------------ | ------------- | -------------
BMAN |    94.45    |       97.23    |         91.32     |            94.23

## Usage

1. download `train.json` and `val.json` from [here](https://thunlp.github.io/fewrel.html)

2. download `glove.6B.50d.json` from [here](https://cloud.tsinghua.edu.cn/f/b14bf0d3c9e04ead9c0a/?dl=1)

3. train model

```
 python train_demo.py --N_for_train 20 --N_for_test 5 --K 1 --Q 5 --batch 1
```


