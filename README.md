# LOTClass

[LOTClass](https://github.com/yumeng5/LOTClass)  的中文实验、学习、应用。提供中文``新闻多分类案例``和数据

- 预备内容：需要 [词汇级BERT](https://github.com/425776024/WoBERT) 

## Requirements

```
$ pip install -r requirements.txt
```

----

## 数据准备
仅提供100条案例数据，见：``datasets/data.csv``
```
python datasets/data_process.py
```


## 配置
指定好WoBERT路径等
```
vim config/configs.yaml

```

## 运行
```shell
python train.py
```