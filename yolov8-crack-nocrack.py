import cv2, os
from ultralytics import YOLO 
from ultralytics.yolo.utils.plotting import Annotator
import argparse

class yolov8_crack_nocrack():
    def __init__(self, model_path):
        self.color = (255, 0, 0)
        self.thickness = 3
        # model_path = r"F:weights\best.pt"
        self.model = YOLO(model_path) #, conf_thres=0.2, iou_thres=0.3)

    def main(self, args):
        # Read image
        store_path = args.crack_path
        image_path = args.image_path
        i = 1
        for filename in os.listdir(image_path):
            print(filename)
            img = cv2.imread(os.path.join(image_path, filename))
            preds = self.model.predict(img)[0]
            classes = preds.boxes.cls
            print('classes - ',classes)
            clss = []

            defects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16]

            for cs in classes:
                print(int(cs))
                if int(cs) in defects:
                    print("defect found")
                    path = store_path
                    final_path = path + '\\' + filename
                    cv2.imwrite(final_path, img)
                    # clss.append(cs)
            print("Image processed - ", i)
            i += 1
            # if len(classes) < 2:
            #     try:
            #         c = preds.boxes.cls
            #         b = preds.boxes.xyxy[0]
            #         # if int(c) == 14:
            #         annotator = Annotator(img)
            #         annotator.box_label(b, model.names[int(c)])
            #         cv2.rectangle(img, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), color, thickness)
            #     except:
            #         pass
            # else:
            #     for r in preds:
            #         # print('r - ',r)
            #         annotator = Annotator(img)
            #         boxes = r.boxes
            #         # print('boxes - ', boxes)
            #         for box in boxes:
                        
            #             b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            #             # print('b - ',b)
            #             c = box.cls
            #             # print('c - ',int(c))
            #             if int(c) != 14:
            #                 annotator.box_label(b, model.names[int(c)])
            #                 cv2.rectangle(img, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), color, thickness)
                    
            # # img = annotator.result()  
            # # cv2.imshow('YOLO V8 Detection', frame)
            # cv2.imwrite(os.path.join(store_path, filename), img)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to trained best.pt file')
    parser.add_argument('--image_path', type=str, help='Path of the image folder')
    parser.add_argument('--crack_path', type=str, help='Path of the crack image folder')
    # parser.add_argument('--threads', type=int, default=100, help='Max threads (only works for --download-images)')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_opt()
    test = yolov8_crack_nocrack(args.model_path)
    test.main(args)

# python yolov8-crack-nocrack.py --model_path F:\weights\best.pt --image_path F:\train3\images --crack_path F:\crack_images
