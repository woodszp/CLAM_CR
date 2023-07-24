# README

#### 环境配置

```
docs/INSTALLATION.md
```

#### train

```bash
python3 train.py train_config.yaml
```

#### test

```bash
python3 test.py test_config.yaml
```

#### 单次测试（特定模型测试特定数据集）

##### 运行

```bash
python3 test_single.py test_single_config.yaml
```

##### 准备

1.将待测试的`.svs`文件放到下面的路径中：

```
CLAM_CRC/test_model/testData
```

> 注：可以是多个测试文件

2.将待测试文件的真实标记信息放到下面的路径中：

```
CLAM_CRC/test_model/test.csv）
```

> 注:可以在`test_single_config.yaml`中修改这个文件的路径

3.将待测模型(.pt文件)，`train.py`训练出的的模型存于下面的路径中：

```
CLAM_CRC/results
```

可将.pt文件移到`CLAM_CRC/test_model`下并改名为`best_model.pt`，或者直接修改配置文件中`test_single_config.yaml`的参数`ckpt_path`来读取待测模型位置。

##### 输出

输出文件存于`CLAM_CRC/test_model/{model_name}.csv`文件中，输出样式如下：

```csv
slide_id,Y,Y_hat,p_0,p_1
Col_PNI2021chall_train_0002.svs,0.0,1.0,0.32284075021743774,0.6771592497825623
```

输出文件从左到右的列分别表示：图像编号（图像名）、真实值、预测值、**预测为0类的概率（自行设置0类的含义）、预测为1类的概率。**

#### 输入

##### 一.病理图像

**1.修改病理图像(.svs文件)读取位置，需修改以下两个配置文件中的参数：**

```
CLAM_CRC/extract_features_config.yaml 
source: /home/webace/Colon_Data

CLAM_CRC/create_patches_config.yaml 
data_slide_dir : /home/webace/Colon_Data
```



##### 二.图像标记文件

**1.标记文件（.csv）示例**

(CLAM_CRC/dataset_csv/label.csv)

```
case_id,slide_id,label
Col_PNI2021chall_train_0002.svs,Col_PNI2021chall_train_0002.svs,subtype_3
Col_PNI2021chall_train_0003.svs,Col_PNI2021chall_train_0003.svs,subtype_3
Col_PNI2021chall_train_0012.svs,Col_PNI2021chall_train_0012.svs,subtype_4
Col_PNI2021chall_test_0001.svs,Col_PNI2021chall_test_0001.svs,subtype_3
Col_PNI2021chall_test_0002.svs,Col_PNI2021chall_test_0002.svs,subtype_4
Col_PNI2021chall_test_0003.svs,Col_PNI2021chall_test_0003.svs,subtype_4
Col_PNI2021chall_test_0004.svs,Col_PNI2021chall_test_0004.svs,subtype_3
Col_PNI2021chall_test_0005.svs,Col_PNI2021chall_test_0005.svs,subtype_4
Col_PNI2021chall_test_0006.svs,Col_PNI2021chall_test_0006.svs,subtype_3
Col_PNI2021chall_test_0007.svs,Col_PNI2021chall_test_0007.svs,subtype_3
Col_PNI2021chall_test_0008.svs,Col_PNI2021chall_test_0008.svs,subtype_4
```



**2.修改标记文件(.csv文件)读取位置，需修改以下三个配置文件中的参数：**

```
CLAM_CRC/create_splits_config.yaml 
csv : dataset_csv/label.csv

CLAM_CRC/train_config.yaml 
csv : dataset_csv/label.csv

CLAM_CRC/test_config.yaml 
csv : dataset_csv/label.csv
```



##### 三.其他配置参数

可根据实际情况对下列配置文件中的参数进行修改

```
#预处理、提取特征相关
CLAM_CRC/create_patches_config.yaml
CLAM_CRC/extract_features_config.yaml  

#训练集、测试集、验证集划分
CLAM_CRC/create_splits_config.yaml 

#训练和测试
CLAM_CRC/train_config.yaml 
CLAM_CRC/test_config.yaml 
```



#### output

**一.提取出的图像特征**

```
CLAM_CRC/output/feat_dir
```



##### 二.训练结果以及生成的最优模型位置(交叉验证)

```
CLAM_CRC/results/exp_sx(x是训练序列号)
```



##### 三.测试结果

```
CLAM_CRC/eval_results
```





# CLAM_CR
