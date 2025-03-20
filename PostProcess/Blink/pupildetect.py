import cv2 
import numpy as np
import os



image = cv2.imread("./EyeTrackData/3/Left/2025-01-05_13-00-46_1505.png")
image = cv2.imread("./EyeTrackData/ZLPTest/Right/2025-01-10_23-54-17_44771.png")


path1 = "D:\EyeTrack\PostProcess\Blink\pupildetect.py"
path2 = "D:\EyeTrack\PostProcess\Models\Haar\haarcascade_eye.xml"
path3 = "D:\EyeTrack\PostProcess\Models\Haar\haarcascade_profileface.xml"
print(os.path.relpath(path2,path1))
eye_cascade = cv2.CascadeClassifier(path3)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = gray.copy()



eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(eyes)

if len(eyes) > 0:
    for (ex, ey, ew, eh) in eyes:
            # 在眼部位置显示红色点阵
            cv2.circle(img, (ex + ew // 2, ey + eh // 2), 5, (0, 0, 255), -1)
            ptA = (ex, ey)
            ptB = (ex + ew, ey + eh)
       

            

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



