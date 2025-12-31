import math
import cv2
import numpy as np
import time
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wcam, hcam = 640, 480   # setting the resolution

pTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

detector = htm.handdect(detectionCon=0.5)

# ---------------------- VOLUME CONTROL SETUP ---------------------- #
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

# Correct way (your code was wrong here)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
# ------------------------------------------------------------------ #

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findpsoe(img)

    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]     # Thumb
        x2, y2 = lmlist[8][1], lmlist[8][2]     # Index finger

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        # distance between thumb and index
        length = math.hypot(x2 - x1, y2 - y1)
        print("Length:", int(length))

        # convert hand distance to volume range
        vol = np.interp(length, [15, 80], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 15:
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (40, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Volume Control", img)
    cv2.waitKey(1)
