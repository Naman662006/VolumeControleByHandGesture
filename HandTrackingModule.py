import cv2
import mediapipe as mp
import time

class handdect():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHand = mp.solutions.hands
        self.Hand = self.mpHand.Hands(self.mode, self.maxHands, int(self.detectionCon), self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.Hand.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handlms, self.mpHand.HAND_CONNECTIONS)

        return img

    def findpsoe(self, img, handNo=0, draw=True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myHands = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                     cv2.circle(img, (cx ,cy), 5, (0,0,255), cv2.FILLED)
        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handdect()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findpsoe(img)
        if len(lmlist) != 0:
            print(lmlist[4])
            print(lmlist[6])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "Fps: " + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
