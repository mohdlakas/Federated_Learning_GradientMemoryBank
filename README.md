# Federated-Learning (PyTorch)
This is still a work in progress and experiments are currently being run. The code and results are not accurate yet and will be updated soon.

Building on the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

Mainly experiments are produced on CIFAR10 (both IID and non-IID). Dirlicht distribution used for sampling data.

Since the purpose of these experiments in paper are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.
The added GMB framework

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the FedAvg experiment with CIFAR10 on CNN using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* To run the FedAvg experiment with CIFAR10 on CNN using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* To run the FedAvg experiment with CIFAR10 on CNN using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):

* To run the same experiment under non-IID condition:


You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

## Results on CIFAR10
#### FedAvg:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |
#### FedProx:
#### SCAFFOLD:
#### GMB:
## Results on CIFAR100
#### FedAvg:

#### FedProx:
#### SCAFFOLD:
#### GMB:

## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
