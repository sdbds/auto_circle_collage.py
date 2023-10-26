import cv2
import random
import numpy as np
from os import path
import os
import glob
from nudenet import NudeDetector

folder_path = "test"
save_path="test2"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

all_labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

expose_labels = [
    "FACE_FEMALE",
    "MALE_BREAST_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
]

cover_labels= [
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths
    
def get_box_by_labels(nudection, labels):
    box = []
    for label in labels:
      output = [item['box'] for item in nudection if item['class'] == label]
      if len(output) != 0:
        box.append(output)
    return box

class Mizutama:
    def __init__(self, img, expose, cover):
        rows, cols, _ = img.shape
        #if max(rows, cols) > 1024:
        #    l = max(rows, cols)
        #    img = cv2.resize(img, (int(cols * 1024 / l), int(rows * 1024 / l)))
        self.expose=expose
        self.cover=cover
        self.img = img
        self.mizutama = []

    def detect_mizugi(self):
        self.mizugi_areas = []
        self.cover_areas = []
        print(f"cover={self.cover}")
        if(len(self.cover)):    
        # 将列表转换为numpy数组
            self.cover_areas = self.cover[0]
        else :
            mzg = cv2.inRange(cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV), np.array([0, 180, 8]), np.array([360, 255, 247]))
            mzg = cv2.erode(mzg, np.ones((1, 1), np.uint8))  # kernel size?
            mzg = cv2.dilate(mzg, np.ones((1, 1), np.uint8)) # kernel size?
            contours, _ = cv2.findContours(mzg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            for contour in contours:
                if contour.size > 25:
                    self.mizugi_areas.append(contour)
    def detect_faces(self):
        #cascade = cv2.CascadeClassifier(path.join(cv2.__file__.replace("__init__.py", "data/haarcascade_frontalface_alt2.xml")))
        #gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        #self.faces = cascade.detectMultiScale(gray)
        self.faces = self.expose[0]
        print(f"face:{self.faces}")
	
    def create_mizutama(self):
        for x, y, w, h in self.faces:
            c = self.create_circle(x + w / 2, y + h / 2, max(h / 4, w / 4))
            if c is not None:
                self.mizutama.append(c)
        #for verts in ([0, 0], [0, self.img.shape[0]], [self.img.shape[1], 0], [self.img.shape[1], self.img.shape[0]]):
        #    c = self.create_circle(verts[0], verts[1])
        #    if c is not None:
        #        self.mizutama.append(c)
        for i in range(0, 8):
            c = self.create_circle(random.randrange(self.img.shape[1]), random.randrange(self.img.shape[0]))
            if c is not None:
                self.mizutama.append(c)

    def create_circle(self, x, y, m = None):
        r = min(self.img.shape[0], self.img.shape[1])
        while self.detect_mizugi_collision(x, y, r):
            r -= 5
            if m is not None and r < m:
                return (x, y, r)
            if r < 20:
                return
        return (x, y, r)

    def detect_mizugi_collision(self, x, y, r):
        if len(self.cover_areas) != 0 :
            for cover_area in self.cover_areas:
                if self.check_rect_collision(cover_area, (x, y, r)):
                    return True
        else :
            for mizugi_area in self.mizugi_areas:
                rect = rect = cv2.boundingRect(mizugi_area)
                if self.check_rect_collision(rect, (x, y, r)):
                    hull = cv2.convexHull(mizugi_area)
                    if self.check_convex_hull_collision(hull, (x, y, r)):
                        return True
        for c in self.mizutama:
            if self.detect_mizutama_collision(c, (x, y, r)):
                return True
        return False

    def detect_mizutama_collision(self, c1, c2):
        # return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 < (c1[2] + c2[2]) ** 2
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 < (c1[2] * 0.9 + c2[2] * 0.9) ** 2

    def check_rect_collision(self, rect, circle):
        rx, ry, rw, rh = rect
        cx, cy, cr = circle
        lr = rx - cr <= cx <= rx + rw + cr and ry <= cy <= ry + rh
        tb = ry - cr <= cy <= ry + rh + cr and rx <= cx <= rx + rw
        tl = (rx - cx) ** 2 + (ry - cy) ** 2 < cr ** 2
        tr = (rx + rw - cx) ** 2 + (ry - cy) ** 2 < cr ** 2
        bl = (rx - cx) ** 2 + (ry + rh - cy) ** 2 < cr ** 2
        br = (rx + rw - cx) ** 2 + (ry + rh - cy) ** 2 < cr ** 2
        return lr or tb or tl or tr or bl or br

    def check_convex_hull_collision(self, hull, circle):
        cx, cy, cr = circle
        if cv2.pointPolygonTest(hull, (cx, cy), False) >= 0.0:
            return True
        for i in range(len(hull)):
            a = hull[i][0]
            b = hull[i + 1][0] if i < len(hull) - 1 else hull[0][0]
            pa = a - (np.array([cx, cy]))
            ab = b - a
            mag = np.linalg.norm(ab)
            if mag < 5:
                continue
            if abs(np.cross(pa, ab) / mag) < cr:
                return True
        return False

    def collage(self):
        self.detect_faces()
        self.detect_mizugi()
        self.create_mizutama()

        # mizutama mask
        mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        for mztm in self.mizutama:
            cv2.circle(mask, (int(mztm[0]), int(mztm[1])), mztm[2], 255, -1)
        img1 = cv2.bitwise_and(self.img, self.img, mask = mask)

        # inpaint and blur
        #blur = cv2.blur(cv2.inpaint(self.img, 255 - mask, 10, cv2.INPAINT_TELEA), (50, 50))
        #img2 = cv2.bitwise_and(blur, blur, mask = 255 - mask)
	
	#创建随机三原色组
        img2 = np.zeros_like(self.img)
        img2[:] = np.random.randint(0, 256, size = 3)      
        img2 = cv2.bitwise_and(img2, img2, mask = 255 - mask)
	
        return cv2.add(img1, img2)


def process_images():
    """处理文件夹中的所有图片并保存结果"""

    FOLDER_PATH = "test"
    SAVE_PATH = "test2"
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

    # 循环遍历文件夹中的所有图片
    for ext in IMAGE_EXTENSIONS:
        for img_path in glob.glob(os.path.join(FOLDER_PATH, "*" + ext)):
            
            nude_detector = NudeDetector()
            nudection = nude_detector.detect(img_path)
            print(nudection)
            expose=get_box_by_labels(nudection,expose_labels)
            cover=get_box_by_labels(nudection,cover_labels)
            # 读取图片
            img = cv2.imread(img_path)

            # 创建一个Mizutama对象
            mztm = Mizutama(img,expose,cover)

            # collage图片
            result = mztm.collage()

            # 以新名称保存结果图片
            image_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(SAVE_PATH, image_name), result)


if __name__ == '__main__':
    process_images()
