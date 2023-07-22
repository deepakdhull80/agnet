# AGNET - age and gender prediction network.
## About
AGNET - age and gender network. In which, we focuses on age estimation, gender classification from still facial images of an individual. We train different models for each problem. We trained 2 CNN based network completely 2 different task.

![alt text](https://github.com/deepakdhull80/agnet/blob/main/test/images/predict/owen-cannon-6TLCSMj8zgE-unsplash.jpg?raw=true)
### Dataset

Here i have used listed datasets to train AGNET.
- UTKFace dataset - https://susanqq.github.io/UTKFace/
- Kaggle Gender dataset - https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
- Kaggle Age dataset - https://www.kaggle.com/datasets/mariafrenti/age-prediction

Reason for being use different dataset's because single dataset not contain different faces which makes dataset less diverse. Diversity is the key, if we want model to perform more generous. 

For gender classification, able to get balanced dataset but struggle to unsure the gender of 1-4 age faces. So i removed the faces of 1-4 age babies.

For age classification data distribution is problem. But still try to curate or reduce the noise of in the dataset manually and removed such categories having very less data size and also reduce the high frequency data to make balance.

### Model

Model architecture second vital key of gender and age classification, even more for realtime job. So, need to be very careful before selecting model because tradeoff there is a tradeoff between accuracy and latency.
Based upon the requirement, we need a less latency in prediction as well as moderate accuracy. 

Pipeline Task for realtime age and gender classification:
- face detection
- gender classification
- age estimation

1. Face Detection(facenet):  Used SOTA model in terms of accuracy and speed.

2. Gender Classification: Trained ResNet34 as a binary classification problem.
    Loss function: we used Binary cross-entropy loss.
    Optimizer: ADAM optimizer with 1e-2 learning rate and having LR Schedular with step_size 3
    Evaluation Metrics: Accuracy is used and able to achieve 97% accuracy on test dataset. So edge cases not able to solve which i have removed from the dataset(person's face age between 1-4). 

3. Age Classification: Trained ResNet34 as a Regression problem in which need to predict the age of a face in between 1-100.
    Loss function: Mean Absolute Error(L1) loss function is used.
    Optimizer: ADAM optimizer with 1e-3 learning rate and having LR Schedular with step_size 3
    Evaluation Metrics: monitoring MAE(mean absolute error) for evaluation and able to achieve 6.8 MAE error.


## Environment Setup:
1. Install python>=3.8, virtualenv and pip.

2. Create new virtual environment and activate it
```bash
    python -m venv env-name
```
```bash
    # linux or ubuntu
    source env-name/bin/activate
    #window
    env-name/Scripts/activate
```

3. Install require packages. Go back to your agnet code base where you will get requirements.txt and run
```bash
    pip install -r requirements.txt
```

## Training
For training need to run this. Few arguments which need to pass
```
    -c : path of config file(config.yaml)
    -vv: [optional] version
    -d: device [default: cpu]
    -fp: file path of csv having these columns [file_path, age, gender]
    -ip: images base path if require otherwise send ''
```

```bash
    python agnet/agnet/train.py \
        -c agnet/agnet/config.yaml \
        -vv=v1 \
        -d=cuda \
        -fp=<csv file> \
        -ip=<image base path>
```

## Inference
For inferencing few tasks we have breakdown like image classification, video's per frame prediction and save it back to in video format(mp4) and live stream prediction through camera.

For inference we have a op.py which is capable of doing all tasks which are mentioned above. It require few arguments
```
    -c : prediction related config file(predict_config.yaml)
    -t : 'video' for video processing, 'image' for image processing, 'camera' for live stream prediction
    -f : [optional] file path if require
    -d : device [default cpu] 
```
### Video processing
```bash
    python ./agnet/op.py -c ./agnet/predict_config.yaml -t video -f test/video/test.mp4 -d cuda
```

### Camera realtime prediction
```bash
    python ./agnet/op.py -c ./agnet/predict_config.yaml -t camera -d cuda
```

### Image prediction
```bash
    python ./agnet/op.py -c ./agnet/predict_config.yaml -t image -f image.jpg -d cuda
```

### Note:

File related task like image or video processing, there output file gonna save in infer/output/*
