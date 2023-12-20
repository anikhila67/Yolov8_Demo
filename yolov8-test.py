import cv2, os
from ultralytics import YOLO 
from ultralytics.yolo.utils.plotting import Annotator

class yolo_detect():
    def __init__(self):
        self.color = (255, 0, 0)
        self.thickness = 3
        # self.model_path = r"F:\visionify\BrightSpot\train3-v9-2\train3\weights\best.pt" 
        self.model_path = r"E:\solar-v9\train6\weights\best.pt"
        self.model = YOLO(self.model_path)

    def detect_image(self, img):
        preds = self.model.predict(img)[0]
        classes = preds.boxes.cls
        print('classes - ',classes)
        defects = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '16']
        if len(classes) < 2:
            try:
                c = preds.boxes.cls
                b = preds.boxes.xyxy[0]
                # if int(c) == 14:
                annotator = Annotator(img)
                annotator.box_label(b, self.model.names[int(c)])
                cv2.rectangle(img, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), self.color, self.thickness)
            except:
                pass
        else:
            for r in preds:
                # print('r - ',r)
                annotator = Annotator(img)
                boxes = r.boxes
                # print('boxes - ', boxes)
                for box in boxes:
                    
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    # print('b - ',b)
                    c = box.cls
                    # print('c - ',int(c))
                    if int(c) != 14:
                        annotator.box_label(b, self.model.names[int(c)])
                        cv2.rectangle(img, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), self.color, self.thickness)
        return img
        

    def detect_batch(self, image):
        return self.detect_image(image)

if __name__ == '__main__':
    yolo = yolo_detect()

    # Read image
    # img_url = '/home/azureuser/nikhil-env/test_images3/3cdf14ecc65d4223b8b6b319f33ac155.tif'

    # # read folder
    store_path = r"C:\Users\nikhi\Pictures\bs-sample2\out-train6"
    image_path = r"C:\Users\nikhi\Pictures\bs-sample2"
    for filename in os.listdir(image_path):
        print(filename)
        img = cv2.imread(os.path.join(image_path, filename))
        img = yolo.detect_image(img)
        cv2.imwrite(os.path.join(store_path, filename), img)
    # 
    # filename = r"F:\visionify\BrightSpot\104263500160.jpg"
    # img = cv2.imread(filename)
    # img = yolo.detect_image(img)
    # cv2.imwrite('panel-image.jpg', img)