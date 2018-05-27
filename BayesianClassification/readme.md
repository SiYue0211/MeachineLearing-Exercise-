## 朴素贝叶斯算法介绍

- 实现： python3.5 + numpy

### 内容
- [垃圾邮件分类](https://github.com/SiYue0211/MeachineLearing-Exercise-/blob/master/BayesianClassification/MailClassification/mailClassify.py)

### 数据集
BayesianClassification-MailClassification-email
      |-- ham 正常邮件
email-|
      |-- spam 垃圾邮件
      
### result
共有样本50个，其中40设置为训练样本，10设置为测试样本，最后测试准确率90%以上。
但是测试样本所有邮件都是英文邮件，如果要处理中文邮件，要涉及中文分词方面的知识。
这部分代码没有涉及。
