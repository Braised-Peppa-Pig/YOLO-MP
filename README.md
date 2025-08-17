# YOLO-MP

Lightweight forest fire detection model. 

## Backbone
- GHGNet
 
When `lightconv=False` in the GHGBlock, the GHGBlock is replaced by the HGBlock. Please note that the HGBlock used here differs slightly from the one in the [Official Library](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py).

- FPSConv
## Dataset
Flame dataset: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs.

The Wildfire Dataset: https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset.

For a detailed introduction to the dataset, please refer to the paper [Flame dataset](https://arxiv.org/pdf/2012.14036v1.pdf) | [The Wildfire Dataset](https://www.mdpi.com/1999-4907/14/9/1697).

Other:

## reference
- https://github.com/ultralytics/ultralytics
