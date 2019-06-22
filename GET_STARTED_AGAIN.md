## GET STARTED AGAGIN

**NOTE**: This code is based on [HRNet-Object-Detecton](https://github.com/HRNet/HRNet-Object-Detection)

### Comparison with original code

- [X] SyncBN: using NVIDIA/apex SyncBN

- [X] Multi-scale training (fixed): training detection models with SyncBN and multi-scale training will crash to terrible results (mAP=0.0 or no boxes are predicted). Fortunatedly, we've investigated a method to solve it by padding input images of different scales(`600*1000`, `800*1333`, `1000*1600`) to a fixed scale `**1000*1600**` and keeping the original aspect ratios.

- [X] Multi-scale training (SimpleDet version): we've implemented multi-scale training strategy used in [SimpleDet](https://github.com/TuSimple/simpledet)

- [X] Multi-node & multi-gpu training: we've tested our code when training with multiple nodes (ONLY on AZURE!). Providing a MASTER IP and PORT. Training without SyncBN will reach normal results while training with SyncBN will fail.