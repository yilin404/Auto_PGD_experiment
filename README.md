# Auto_PGD_experiment

## 训练网络

### ResNet

```python
python ./resnet_train.py --dataset [your_dataset] --batch_size [your_batch_size] --num_epochs [your_num_epochs] --model [resnet18 or resnet34 or resnet50]
```

### Inception_v3

```python
python ./inception_train.py --batch_size [your_batch_size] --num_epochs [your_num_epochs]
```

### MobileNet

```python
python ./mobilenet_train.py --batch_size [your_batch_size] --num_epochs [your_num_epochs]
```

生成的结果将保存在./model/{model}文件夹下

## 攻击网络

```python
python ./autopgd_attack.py --dataset [your_dataset] --model [resnet18 or resnet34 or resnet50]
```

## 生成对抗样本和噪声

```python
python ./generate_fake.py --dataset [your_dataset] --model [resnet18 or resnet34 or resnet50]
```

## 黑盒攻击

```python
python blackbox_attack.py --model [Inception or MobileNet or resnet18]
```
