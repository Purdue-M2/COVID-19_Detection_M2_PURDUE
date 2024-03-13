# Robust COVID-19 Detection in CT Image with CLIP

### 1. [Data is provided by the 4th COV19D Competition](https://mlearn.lincoln.ac.uk/ai-mia-cov19d-competition/), from [paper](https://arxiv.org/pdf/2403.02192v2.pdf) 

### 2. Put the 'data' folder under the Challenge3 directory, use [CLIP ViT L/14](https://github.com/openai/CLIP) to extract image features and save them into h5 file (e.g., train_clip.h5 and val_clip.h5) 

### 3. Train the model 
#### Supervised Learning
* load 'data/train_clip.h5', 'data/train.txt' for train_dataset in [train.py](./train.py); load 'data/val_clip.h5', 'data/val.txt' for val_dataset in [train.py](./train.py).

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

```python
python train.py
```

* Use CVaR

```python
model_trainer(loss_type='dag', batch_size=32, num_epochs=32)
```
Tune **alpha_cvar** in dag loss to find the best hyperparameter. 

* Use CVaR with AUC loss

After getting the best alpha_cvar, tune **alpha**, the weight trade-off between cvar and auc. There are also **gamma** and **p** that need to be tuned.
```python
model_trainer(loss_type='auc', batch_size=32, num_epochs=32)
```
#### Semi-Supervised Learning
* Train teacher model
  * Load 'data/all_train_teach.h5', 'data/combined_train.txt' for train_dataset in [train.py](./train.py); load 'data/task2_val_clip_vit.h5', 'data/task2_val.txt' for val_dataset in [train.py](./train.py).
  * Tune the hyperparameter as Task1.
  * The checkpoints will be saved in 'checkpoints_teach/' folder.

* Train student model
  * First run [test.py](./test.py) to give predicted labels to the nonannotated subdataset. After finishing running, the labels will be saved in 'test_predictions.txt'.
    ```python
    python test.py
    ```
  * Secondly, run the [get_path.py](./get_path.py) to combine the nonannotated image path and its corresponding label.
  * Creat2 a new txt 'all_train_stu.txt', which contains annotated labels and predicted labels in step2.
  * Load 'data/all_train_stu.h5', 'data/all_train_stu.txt' for train_dataset in [train.py](./train.py); load 'data/task2_val_clip_vit.h5', 'data/task2_val.txt' for val_dataset in [train.py](./train.py).
  * Train the student model by running

  ```python
  python train.py
  ```
  *  Tune the hyperparameter while training the student model.
  

