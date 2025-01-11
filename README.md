# Super-Resolution Assisted Dual-Branch YOLO for Enhanced Small Target Detection in Remote Sensing Images
Welcome to the repository for Super-Resolution Assisted Dual-Branch YOLO, an advanced algorithm designed to enhance small target detection in remote sensing images.
The code in this repository provides data support for the paper "Super-Resolution Assisted Dual-Branch YOLO for Enhanced Small Target Detection in Remote Sensing Images", which has been published on the journal "The Visual Computer".
![image](https://github.com/user-attachments/assets/0988a43d-8056-4dcc-bb40-9483b5c71ebf)

# Dependencies

 base ----------------------------------------
 
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow
PyYAML>=5.3.1
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0

 logging -------------------------------------
 
tensorboard>=2.4.1
 wandb

 plotting ------------------------------------
 
seaborn>=0.11.0
pandas

 export --------------------------------------
 
 coremltools>=4.1
 onnx>=1.8.1
 scikit-learn==0.19.2  # for coreml quantization

 extras --------------------------------------
 
thop==0.0.31.post2005241907  # FLOPS computation
pycocotools>=2.0  # COCO mAP

results--------------------------------------

xlsxwriter>=3.0.1

# Download Datasets
Download datasets from the [baiduyun](https://pan.baidu.com/s/1L0SWi5AQA6ZK9jDIWRY7Fg) (code: hvi4) links and place them in this directory.

# important file and document instructions
transform_vedai.py:Dataset Processing
test.py:Inference and test of the ASR_YOLO Model
train.py:Training the ASR_YOLO Model
models:Store the source code of models for comparison.
