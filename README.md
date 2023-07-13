# YOLOv1
My implementation of YOLO v1
 
## Usage
### Training
You should downloads pretrained wheights of Yolo architecture to reduce the amount of time required for the convergence and thus ease your burden on the training.
Then chose your dataset and put it a folder named `dataset`.

```bash
python3 train.py workdir
```
`workdir` is `.` by default.

### Decoder
You can use your model or just load a trained model.
If you want to load another trained model put the weights in `checkpoints` directory and name it `best`.