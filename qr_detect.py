import cv2
from pyzbar import pyzbar
import requests
import numpy as np


def line_cross(line1, line2):
    cross_p = None
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0

    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            cross_p = [x,y]
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
        cross_p = [x,y]
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        cross_p = [x,y]
    return cross_p


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    openimg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)
    binImg = cv2.adaptiveThreshold(openimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 0)
    qr_detect = []
    qrs = pyzbar.decode(binImg)
    for qr in qrs:
        code = qr.data.decode('utf-8')
        pts = qr.polygon
        l1 = (pts[0].x, pts[0].y, pts[2].x, pts[2].y)
        l2 = (pts[1].x, pts[1].y, pts[3].x, pts[3].y)
        cp = line_cross(l1,l2)
        if not cp is None:
            qr_detect.append({'cx':cp[0],'cy':cp[1],'label':code})
    return qr_detect


ports = [8080,8081]
for port in ports:
    try:
        resp = requests.get('http://192.168.0.88:{}/?action=snapshot'.format(port),timeout=2)
        if resp.ok:
            continue
        # f = open('{}.jpg'.format(port),'rb')
        # data = f.read()
        data = resp.content
        img = cv2.imdecode(np.array(bytearray(data)),cv2.IMREAD_COLOR)
        qr_resp = detect(img)
        print(qr_resp)
    except Exception as e:
        print(e)
        
