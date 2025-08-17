import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\zhuho\Desktop\投稿\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'C:\Users\zhuho\Desktop\ultralytics\ultralytics-20241109\ultralytics-main\dataset\data.yaml',
                cache=False,
                imgsz= 640,
                epochs= 350,
                batch= 12,
                close_mosaic= 0,
                workers= 4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project=r'C:\Users\zhuho\Desktop\投稿\ultralytics-main\runs',
                name='yolov8n',
                )