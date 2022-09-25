# Head Pose Estimation using FSA Net
Scientific research 2022 about Head Pose Estimation using modified FSA Net

This work is based on:
+ The [FSA Net repo](https://github.com/shamangary/FSA-Net) and [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_FSA-Net_Learning_Fine-Grained_Structure_Aggregation_for_Head_Pose_Estimation_From_CVPR_2019_paper.pdf) of Yang et al.
+ A third-party Pytorch implementation github [repo](https://github.com/omasaht/headpose-fsanet-pytorch)
## Demo
Video file or a camera index can be provided to demo script. If no argument is provided, default camera index is used.
### Video file usage
For any video format that OpenCV supported (mp4, avi etc.): 
```
python3 demo.py --video /path/to/video.mp4
```
### Camera usage
```
python3 demo.py --cam 0
```
