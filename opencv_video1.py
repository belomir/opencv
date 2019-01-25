#!/usr/bin/env python3

import cv2 as cv
import numpy as np

OUT = False

cap = cv.VideoCapture(0)
if OUT:
    out = cv.VideoWriter("out.avi", cv.VideoWriter_fourcc(*"XVID"), 7.5, (640, 480))

delta = 20
color = 120 #60
low = (color-delta, 150, 100)
high = (color+delta, 255, 255)

try:
    while cap.isOpened():
        ret, frm = cap.read()
        frm = cv.flip(frm, 1)
        orig = frm.copy()
        frm = cv.cvtColor(frm, cv.COLOR_RGB2HSV)
        frm = cv.inRange(frm, low, high)
        _, C, H = cv.findContours(frm, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    
        C = list(filter(lambda x: cv.contourArea(x)>1000, C))
        if len(C)>0:
            r = cv.minAreaRect(C[0])
            b = cv.boxPoints(r)
            b = np.int0(b)
            b0 = b.tolist()
            lines = [[b0[0], b0[1]], [b0[1], b0[2]]]
            lens = list(map(lambda l: (l[0][0]-l[1][0])**2+(l[0][1]-l[1][1])**2, lines))
            l = lines[lens.index(max(lens))]
            x = l[0][0]-l[1][0]
            y = l[0][1]-l[1][1]
            z = int(np.sqrt(x**2+y**2))
            a = np.arccos(abs(x)/z)
            a = int(a/np.pi*180)
            a = 180-a if x>0 else a
            M = cv.moments(b)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv.drawContours(orig, [b], -1, (0, 255, 0), 2)
            cv.putText(orig, "{0}".format(str(a)), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv.imshow("1", orig)
        if OUT:
            out.write(orig)
        if cv.waitKey(1) != 255:
            break
finally:
    cap.release()
    if OUT:
        out.release()
    cv.destroyAllWindows()
