# Homework for Introduction to Machine Learning

## Homework 1: Naive Bayes

### 代码结构

代码主要有：

- `./main.py` 主程序，负责执行相关程序
- `./data.py` 负责处理/读入数据
	- *method* `chinese_email_data_set(path)` 本次实验所用的数据
	- *class* `Dictionary` 一个类，用来整理Corpus的单词，并把单词的字符串转成index。
	- *method* `shuffle_data(full_data, split)` 用来对数据集做random shuffle，并按照比例来划分成training/test dataset。
- `./model.py` 主要负责实现的程序
	- *class* `NaiveBayes` 本次实验的模型，在构造函数的时候传入training set和字典，即进行estimate相关参数，从`query(data)`方法中可以给出预测（返回spam的概率）
- `./evaluation.py` 主要负责评估结果：
	- *method* `binary_classifier(model, test)` 对于二分类问题，返回accuracy, precision, recall, F1 score，其中model需要有`query(data)`方法，`test`是测试集合，`test[i][0]`是第i个数据的label，必须是0/1，`test[i][1]`是第i个数据的feature，需要喝`model`的`query(data)`方法传的`data`一致。

### 运行

需要把`data/`和`data_cut/`文件放在文件夹`./trec06c-utf8/`下

需要安装numpy。

运行`python main.py`即可使用8:2的划分，使用全部training set的数据，运行5次，并得到相关的评估指标的值。


