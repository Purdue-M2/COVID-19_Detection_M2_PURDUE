# Robust COVID-19 Detection in CT Images with CLIP

Li Lin, Yamini Sri Krubha, Zhenhuan Yang, Cheng Ren, Xin Wang, Shu Hu
_________________

This repository is the official implementation of our paper [Robust COVID-19 Detection in CT Images with CLIP](https://arxiv.org/abs/2403.08947).

### 1. Data Preparation
* [Data is provided by the 4th COV19D Competition](https://mlearn.lincoln.ac.uk/ai-mia-cov19d-competition/), from [paper](https://arxiv.org/pdf/2403.02192v2.pdf). 

* After getting the Covid-19 CT scan data, use [CLIP ViT L/14](https://github.com/openai/CLIP) to extract image features and save them into h5 file (e.g., train_clip.h5 and val_clip.h5) by executing [clip_feature.py](./clip_feature.py).
Note: medical images, such as CT scans, are one-channel images; in [clip_feature.py](./clip_feature.py), we convert them to 3 channels; all 3 channels are the same value.
```python
python clip_feature.py
```

* Put the processed h5 file under the data directory.

### 2. Train the model 
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

#### Semi-Supervised Learning
* Train teacher model
  * Load 'data/all_train_teach.h5', 'data/combined_train.txt' for train_dataset in [train.py](./train.py); load 'data/task2_val_clip_vit.h5', 'data/task2_val.txt' for val_dataset in [train.py](./train.py).
  * Tune the hyperparameter as in supervised learning.
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

**Note**: Metrics like ACC and F1 scores recorded during training are calculated on the image sample level. To produce the same results as reported in the paper, you should take into account that there exists only a single label for the whole CT scan and no labels for each CT scan image.
    
### 3. Predict on test data
Execute [predict_CT_scans.py](./predict_CT_scans.py) will generate a CSV file that contains CT scan folder name and its label.
```python
python predict_CT_scans.py
```
## Citation
Please kindly consider citing our papers in your publications. 
```bash
@article{lin2024robust,
      title={Robust COVID-19 Detection in CT Images with CLIP}, 
      author={Li Lin and Yamini Sri Krubha and Zhenhuan Yang and Cheng Ren and Xin Wang and Shu Hu},
      year={2024},
      eprint={2403.08947},
      archivePrefix={arXiv},
}
```
  

