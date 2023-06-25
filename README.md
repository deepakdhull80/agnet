# agnet

## About

## Dataset

## Training

```bash
    !python agnet/agnet/train.py \
        -c agnet/agnet/config.yaml \
        -vv=v1 \
        -d=cuda \
        -fp=<csv file> \
        -ip=<image base path>
```


## Inference
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