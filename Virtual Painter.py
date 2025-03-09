import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

##################
brushThickness = 15
eraserThickness = 75

##################

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)

# importing images
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

# running webcam
header = overlayList[0]
drawColor = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:
    # 1) Import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2) Find hand Landmarks
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) >= 12:  # Ensure at least 12 landmarks exist

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3) Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        #4)If selection mode - two fingers are up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            print("selection mode")
            #checking for click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        #5)If drawing mode - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv) #color image on black canvas and black image on original canvas
    img = cv2.bitwise_or(img, imgCanvas) #color image on original canvas by or operation

    # setting the header image
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) #merged original canvas with the other one
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
