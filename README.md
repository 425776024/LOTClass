# LOTClass

[LOTClass](https://github.com/yumeng5/LOTClass)  的中文实验、学习、应用。提供中文``新闻多分类案例``和数据

- 预备内容：需要 [词汇级BERT](https://github.com/425776024/WoBERT)   ，这是一个中文、Pytorch的词汇级别embedding的BERT模型

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

- 会进行3步，其中第一步生成类别词汇，类别为：``datasets/label_names.txt``
- 类别词汇类似：``datasets/category_vocab.txt``是自己数据生成的，供参考，完整训练数据需要你自己准备