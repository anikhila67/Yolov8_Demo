from ultralytics import YOLO
import wandb
import argparse
wandb.init()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to pretrained file')
    parser.add_argument('--dataset_path', type=str, help='Path of the dataset folder')
    parser.add_argument('--epochs', type=str, help='Number of epochs')
    # parser.add_argument('--threads', type=int, default=100, help='Max threads (only works for --download-images)')
    opt = parser.parse_args()
    return opt

def train(args):
    yolo_model = args.model_path
    dataset_path = args.dataset_path
    epochs = args.epochs
    model = YOLO('yolov8n.pt')
    model.train(data = "/home/azureuser/nikhil-env/solar-dataset-v9/data.yaml", epochs=80)

# yolo task=detect mode=predict model=yolov8n.pt source='/home/azureuser/nikhil-env/yolov5/solar-dataset-v6-noaug-yolo/train/images/9630e155bb1a412eb54d2bc95e88ce11.tif'

if __name__ == '__main__':
    args = parse_opt()
    train(args)
