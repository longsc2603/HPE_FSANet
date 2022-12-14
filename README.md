# Head Pose Estimation using FSA Net
Scientific research 2022 about Head Pose Estimation using modified FSA Net
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
## Training and testing
### 1. Original FSANet:
For training, checkout the notebook: **src/2-Train Model.ipynb**.

For testing, checkout the notebook: **src/2-Test Model.ipynb**.

I make two Python files from those notebooks named **src/train_fsa.py** and **src/test_fsa.py** in case you want to run locally instead of using .ipynb files on Google Colab.
### 2. FSANet combined with Triplet Network architecture:
Basically, everything is the same as in part 1, but please use the modified files in folder src_triplet/ instead of src/
## Dataset
For model training and testing, you can download the preprocessed dataset from author's official git repository and place them inside the data/ directory. Your dataset hierarchy should look like this:
```
data/
  type1/
    test/
      AFLW2000.npz
    train/
      AFW.npz
      AFW_Flip.npz
      HELEN.npz
      HELEN_Flip.npz
      IBUG.npz
      IBUG_Flip.npz
      LFPW.npz
      LFPW_Flip.npz
```
## Acknowledgements
This work is based on:
+ The [FSA Net repo](https://github.com/shamangary/FSA-Net) and [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_FSA-Net_Learning_Fine-Grained_Structure_Aggregation_for_Head_Pose_Estimation_From_CVPR_2019_paper.pdf) of Yang et al.
+ A third-party Pytorch implementation github [repo](https://github.com/omasaht/headpose-fsanet-pytorch) (This is where all the files are from, some of them are modified for training FSANet with the Triplet Network architecture)
