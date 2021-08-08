from dataclasses import dataclass
from typing import List, Union
import numpy as np
import cv2
import itertools as it
from shapely.geometry import MultiLineString, LineString, Point, MultiPoint


@dataclass
class Ellipse():
    x: float = None # center x pos
    y: float = None # center y pos
    a: float = None # semi-major axis
    b: float = None # semi-minor axis
    ang: float = None # rotation angle

    def to_cv2(self):
        return ((self.x, self.y),(self.a,self.b), self.ang)


@dataclass
class BullsEye():
    x: float = None
    y: float = None

    def xy(self):
        return (self.x, self.y)


class Calibration():

    def __init__(self, img) -> None:
        self.image = img
        self.cal_ellipse = Ellipse()
        self.bullseye = BullsEye()

    def preprocess_ellipse(self) -> np.ndarray:
        # convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # use saturation channel for binary thresholding
        ret, thresh = cv2.threshold(hsv[:,:,1], 150, 255, cv2.THRESH_BINARY)
        # remove radial wires
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
        calib_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return calib_img

    def ellipse_mask(self, image: np.ndarray) -> np.ndarray:
        assert self.cal_ellipse.x, "No ellipse detected. Do the ellipse detection first."
        if len(image.shape) > 2:
            H, W, C = image.shape
        else:
            H, W = image.shape
            C = None
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        cost = np.cos(np.deg2rad(self.cal_ellipse.ang))
        sint = np.sin(np.deg2rad(self.cal_ellipse.ang))
        r = (((x-self.cal_ellipse.x)*cost+(y-self.cal_ellipse.y)*sint)**2/(self.cal_ellipse.a/2)**2)+\
            (((x-self.cal_ellipse.x)*sint+(y-self.cal_ellipse.y)*cost)**2/(self.cal_ellipse.b/2)**2)
        mask = r <= 1
        if C:
            image[np.where(~mask)] = np.zeros(C, dtype=int)
        else:
            image[np.where(~mask)] = 0
        return image

    def find_ellipse(self) -> List[Ellipse]:
        # preprocess image
        calib_img = self.preprocess_ellipse()
        # find contours
        contours, hierarchy = cv2.findContours(calib_img, 1, 2)
        # fit ellipses
        ellipses = []
        areas = []
        for cnt in contours:
            try:
                ell = cv2.fitEllipse(cnt)
                # valid ellipses have centers inside the image
                if (0 < ell[0][0] < self.image.shape[0]) and (0 < ell[0][1] < self.image.shape[1]):
                    ellipses.append(ell)
                    areas.append(cv2.contourArea(cnt))
            except:
                pass
        if len(ellipses) == 0:
            raise("No calibration ellipse found. Please check the webcam.")
        # set ellipse with largest area as calibration ellipse
        ellipse = ellipses[np.argmax(areas)]
        self.cal_ellipse = Ellipse(
                            x=ellipse[0][0], y=ellipse[0][1],
                            a=ellipse[1][0], b=ellipse[1][1],
                            ang=ellipse[2]
                            )

    def preprocess_wires(self) -> np.ndarray:
        # convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # use value channel for binary thresholding
        ret, thresh = cv2.threshold(hsv[:,:,2], 150, 255, cv2.THRESH_BINARY)
        # mask area outside the dartsboard ellipses
        thresh = self.ellipse_mask(thresh)
        # apply Canny edge detection
        canny = cv2.Canny(thresh,100,200)
        return canny

    def find_bullseye(self, return_lines=False) -> Union[None, MultiLineString]:
        # preprocess image
        canny = self.preprocess_wires()
        # Hough line detection
        lines = cv2.HoughLines(canny, rho=2, theta=np.pi / 80, threshold=100)
        # find all intersection points of the 10 most important lines
        maxshape = np.max(self.image.shape)
        l = []
        for dist, angle in lines[:10,0]:
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            dir_v = np.array([np.cos(angle+np.pi/2), np.sin(angle+np.pi/2)])
            xs, ys = np.array([x0, y0]) - dir_v*maxshape
            xe, ye = np.array([x0, y0]) + dir_v*maxshape
            l.append(LineString([(xs,ys),(xe,ye)]))
        m = MultiLineString(l)
        # make point cloud with intersections
        p = []
        for i, j in it.combinations(np.arange(10), 2):
            if isinstance(m[i].intersection(m[j]), Point):
                p.append(m[i].intersection(m[j]))
        # Bull's eye position is the centroid of this point cloud
        c = MultiPoint(p).centroid.coords.xy
        self.bullseye.x = c[0][0]
        self.bullseye.y = c[1][0]
        if return_lines:
            return m

    def transform_image(self):
        assert self.cal_ellipse, "Find ellipse first."
        cos_t = np.cos(np.deg2rad(self.cal_ellipse.ang))
        sin_t = np.sin(np.deg2rad(self.cal_ellipse.ang))
        #x = self.cal_ellipse.x
        x = self.bullseye.x
        #y = self.cal_ellipse.y
        y = self.bullseye.y
        a = self.cal_ellipse.a
        b = self.cal_ellipse.b

        R1 = np.array([ [cos_t,  sin_t,   0],
                        [-sin_t, cos_t,   0],
                        [0,      0,       1]])
        R2 = np.array([ [cos_t, -sin_t,   0],
                        [sin_t,  cos_t,   0],
                        [0,      0,       1]])
        T1 = np.array([ [1,      0,      -x],
                        [0,      1,      -y],
                        [0,      0,       1]])
        T2 = np.array([ [1,      0,       x],
                        [0,      1,       y],
                        [0,      0,       1]])
        D  = np.array([ [1,      0,       0],
                        [0,      a/b,     0],
                        [0,      0,       1]])

        M = T2@R2@D@R1@T1

        return M

    def do(self) -> None:
        self.find_ellipse()
        self.find_bullseye()

        return self