# Challenge3

### 1. [download processed data](https://purdue0-my.sharepoint.com/:u:/g/personal/lin1785_purdue_edu/EYUCs8e0GjtKqkttn3mhEJUBwE5SWi9Nxk4288twnGJw9Q?e=HcaPv6)

### 2. Put the 'data' folder under the Challenge3 directory.

### 3. Train the model 
#### Task1
* load 'data/train_clip.h5', 'data/train.txt' for train_dataset in [train.py](./train.py); load 'data/val_clip.h5', 'data/val.txt' for val_dataset in [train.py](./train.py);

```python
    train_dataset = UniAttackDataset(
    hdf5_filename='data/train_clip.h5',
    labels_filename='data/train.txt',
    dataset_name='train_features'
)
    val_dataset = UniAttackDataset(
    hdf5_filename='data/val_clip.h5',
    labels_filename='data/val.txt',
    dataset_name='val_features'
)
```

* Use CVaR

```python
model_trainer(loss_type='dag', batch_size=32, num_epochs=32)
```

* Use CVaR with AUC loss

```python
model_trainer(loss_type='auc', batch_size=32, num_epochs=32)
```
