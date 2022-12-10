import sys
import cv2
from models.layers import BBoxRegressor
from models.change_classifier import ChangeClassifier
import torch

def showBoxes(generated_boxes, label_boxes, mask):
    
    generated_boxes = generated_boxes[0]
    label_boxes = label_boxes[0]

    i = 0
    while i < (len(generated_boxes)):
        if generated_boxes[i:i+4].sum() == 0:
            break
        else:
            x = generated_boxes[i].long().item()
            y = generated_boxes[i+1].long().item()
            w = generated_boxes[i+2].long().item()
            h = generated_boxes[i+3].long().item()
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0,255,0), 4)
            i = i+4

    i = 0
    while i < len(label_boxes):
        if label_boxes[i:i+4].sum() == 0:
            break
        else:
            x = label_boxes[i].long().item()
            y = label_boxes[i+1].long().item()
            w = label_boxes[i+2].long().item()
            h = label_boxes[i+3].long().item()
            cv2.rectangle(mask, (x, y), (x+w, y+h), (0,0,255), 4)
            i = i+4

    cv2.imwrite("test.png", mask)

def calculateAccuracyScore(generated_boxes, label_boxes):
    generated_boxes = generated_boxes[0]
    label_boxes = label_boxes[0]

    i = 0
    accuracy_score = 0
    while i < (len(generated_boxes)):
        if generated_boxes[i:i+4].sum() == 0:
            break
        else:
            x1 = generated_boxes[i].long().item()
            y1 = generated_boxes[i+1].long().item()
            w1 = generated_boxes[i+2].long().item()
            h1 = generated_boxes[i+3].long().item()

            x2 = label_boxes[i].long().item()
            y2 = label_boxes[i+1].long().item()
            w2 = label_boxes[i+2].long().item()
            h2 = label_boxes[i+3].long().item()
            
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = max(x1 + w1, x2 + w2)
            x_left = max(y1 + h1, y2 + h2)

            if (x_right < x_left or y_bottom < y_top):
                return 0

            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
            bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            accuracy_score += iou
            i = i+4

    if (i == 0 and label_boxes.sum() != 0):
        return 0
    elif (i == 0 and label_boxes.sum() == 0):
        return 1
    else:
        return (accuracy_score/(i/4))


def main():
    filename = sys.argv[1]
    modelname = sys.argv[2]

    #get mask, testImg, refImg
    mask_filename = "LEVIR-CD-256/label/" + filename
    test_filename = "LEVIR-CD-256/A/" + filename
    ref_filename = "LEVIR-CD-256/B/" + filename

    testImg = cv2.imread(test_filename, cv2.IMREAD_COLOR)
    refImg = cv2.imread(ref_filename, cv2.IMREAD_COLOR)

    testTensor = torch.tensor(testImg.reshape(1, 3, 256, 256)).float()
    refTensor = torch.tensor(refImg.reshape(1, 3, 256, 256)).float()

    mask = cv2.imread(mask_filename, cv2.IMREAD_COLOR)
    copy = mask.copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    bbox_regressor = BBoxRegressor()
    label_boxes = bbox_regressor.get_bboxes(torch.tensor(mask).reshape(1, 256, 256), 50)

    model = ChangeClassifier()
    model.load_state_dict(torch.load(modelname))

    generated_boxes = model(refTensor, testTensor)

    accuracy = calculateAccuracyScore(generated_boxes, label_boxes)
    print("accuracy_score: ", accuracy)
    showBoxes(generated_boxes, label_boxes, copy)
    return

if __name__ == "__main__":
    main()
