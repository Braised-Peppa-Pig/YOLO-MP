# YOLO-MP

Lightweight forest fire detection model.

## Environment
For the required environment, run `
pip install -r requirements.txt`

## Train
- train.py: Configure your dataset path and other  hyperparameters in `train.py` for training
- CLI: Run yolo train `model=yolov8n.pt/$.yaml data=$.yaml epochs=350 imgsz=640 batch=16` for training
## Predict
```shell:copy
yolo predict model=$.pt source='../fire.jpg'
```
## Dataset
Flame dataset: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs.

The Wildfire Dataset: https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset.

For a detailed introduction to the dataset, please refer to the paper [Flame dataset](https://arxiv.org/pdf/2012.14036v1.pdf) | [The Wildfire Dataset](https://www.mdpi.com/1999-4907/14/9/1697).


## reference
- https://github.com/ultralytics/ultralytics
