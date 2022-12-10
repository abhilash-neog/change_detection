import sys
import cv2
import numpy as np

def main():
    filename = sys.argv[1]
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    result = image.copy()
    for cntr in contours:
        pad = 0
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x-pad, y-pad), (x+w+pad, y+w+pad), (0,0,255), 4)

    cv2.imwrite("boundingBox_Image.png", result)
    return

def trainsform_filename(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    result = image.copy()
    boxes = np.array([])
    for cntr in contours:
        pad = 0
        x,y,w,h = cv2.boundingRect(cntr)
	boxes.append([x,y,w,h])

    return boxes

if __name__=="__main__":
    main()
