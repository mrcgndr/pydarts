from dataclasses import dataclass
from typing import List, Union, Tuple
import numpy as np
import cv2
import itertools as it
from shapely.geometry import MultiLineString, LineString, Point, MultiPoint
from scipy.optimize import minimize
from util import rotate_anchor




@dataclass
class Ellipse():
    x: float = None # center x pos
    y: float = None # center y pos
    b: float = None # semi-major axis
    a: float = None # semi-minor axis
    ang: float = None # rotation angle

    def to_cv2(self) -> Tuple[Union[Tuple, float]]:
        return ((self.x, self.y),(self.b,self.a), self.ang)

    def area(self) -> float:
        return np.pi*self.a*self.b

    def get_edge_points(self) -> np.ndarray:
        a = np.deg2rad(self.ang)
        return np.array([[self.x-np.cos(a)*self.b/2, self.y-np.sin(a)*self.b/2],  # top
                         [self.x+np.sin(a)*self.a/2, self.y-np.cos(a)*self.a/2],  # right
                         [self.x+np.cos(a)*self.b/2, self.y+np.sin(a)*self.b/2],  # bottom
                         [self.x-np.sin(a)*self.a/2, self.y+np.cos(a)*self.a/2]]) # left

    def transform_edge_points(self, M: np.ndarray) -> np.array:
        t = (M@np.column_stack((self.get_edge_points(), np.ones(4))).T).T
        t /= t[:,2,None]
        return t[:,:2]

    def get_equalize_rotation_matrix(self) -> np.ndarray:
        return rotate_anchor(gamma=-self.ang, x=self.x, y=self.y)

    def get_ellipse_to_circle_rotation_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0], [0, self.b/self.a, 0], [0, 0, 1]])


@dataclass
class BullsEye():
    x: float = None
    y: float = None

    def xy(self) -> Tuple[float]:
        return (self.x, self.y)

    def transform(self, M: np.ndarray) -> np.array:
        t = (M@np.array([self.x, self.y, 1])[:,None])[:,0]
        t /= t[2]
        return t[:2]


class Calibration():

    def __init__(self, img) -> None:
        self.image = img
        if len(img.shape) > 2:
            self.h, self.w, self.c = img.shape
        else:
            self.h, self.w = img.shape
            self.c = None
        self.ellipse = Ellipse()
        self.bullseye = BullsEye()

    def preprocess_ellipse(self) -> np.ndarray:
        # convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # use saturation channel for binary thresholding
        _, thresh = cv2.threshold(hsv[:,:,1], 120, 255, cv2.THRESH_BINARY)
        # remove radial wires
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
        calib_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return calib_img

    def ellipse_mask(self, image: np.ndarray) -> np.ndarray:
        assert self.ellipse.x, "No ellipse detected. Do the ellipse detection first."
        x, y = np.meshgrid(np.arange(self.w), np.arange(self.h))
        cost = np.cos(np.deg2rad(self.ellipse.ang))
        sint = np.sin(np.deg2rad(self.ellipse.ang))
        # ellipsis equation
        r = (((x-self.ellipse.x)*cost+(y-self.ellipse.y)*sint)**2/(self.ellipse.b/2)**2)+\
            (((x-self.ellipse.x)*sint+(y-self.ellipse.y)*cost)**2/(self.ellipse.a/2)**2)
        mask = r <= 1
        if len(image.shape) > 2:
            image[np.where(~mask)] = np.zeros(image.shape[2], dtype=int)
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
        self.ellipse = Ellipse(
                            x=ellipse[0][0], y=ellipse[0][1],
                            b=ellipse[1][0], a=ellipse[1][1],
                            ang=ellipse[2]
                            )

    def preprocess_wires(self) -> np.ndarray:
        # convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # use value channel for binary thresholding
        ret, thresh = cv2.threshold(hsv[:,:,2], 120, 255, cv2.THRESH_BINARY)
        # mask area outside the dartsboard ellipses
        thresh = self.ellipse_mask(thresh)
        # apply Canny edge detection
        canny = cv2.Canny(thresh,100,200)
        return canny

    def find_bullseye(self, return_lines=False) -> Union[None, MultiLineString]:
        # preprocess image
        canny = self.preprocess_wires()
        # Hough line detection
        lines = cv2.HoughLines(canny, rho=2, theta=2*np.pi/360, threshold=100)
        # find all intersection points of the 10 most important lines
        maxshape = np.max(self.image.shape)
        l = []
        angles = []
        for dist, angle in lines[:10,0]:
            angles.append(angle)
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
            return m, angles

    def __opt_rotation(self, phi: float, bullseye_rot: np.array, M1: np.ndarray) -> float:
        M2 = rotate_anchor(phi=phi, x=bullseye_rot[0], y=bullseye_rot[1])
        ell_edge_rot = self.ellipse.transform_edge_points(M2@M1)
        # calculate distances from Bull's Eye to left and right ellipse edges
        dists = np.sqrt(np.sum((ell_edge_rot[[0,2]]-bullseye_rot)**2, axis=1))
        # loss = squared difference between distances
        return np.diff(dists)[0]**2

    def transform_to_circle(self) -> np.ndarray:
        assert self.ellipse.x, "Find ellipse first."
        assert self.bullseye.x, "Find Bull's Eye first."
        # Step 1 - equalize ellipsis rotation
        M1 = self.ellipse.get_equalize_rotation_matrix()
        # Step 2 - rotate ellipse, so that Bull's Eye is in the ellipse center calculate new bulleye coordinates and ellipse edge points
        bullseye_rot = self.bullseye.transform(M1)
        #   vectorize loss function
        opt_func = np.vectorize(lambda phi: self.__opt_rotation(phi, bullseye_rot=bullseye_rot, M1=M1))
        res = minimize(opt_func, x0=0, bounds=[(-0.05,0.05)])
        print(res)
        if not res.success:
            raise "Cannot rotate dartsboard correctly."
        #   result = best phi for rotation matrix M2
        M2 = rotate_anchor(phi=res.x[0], x=bullseye_rot[0], y=bullseye_rot[1])
        # Step 3 - transform ellipse to circle
        M3 = self.ellipse.get_ellipse_to_circle_rotation_matrix()

        return M3@M2@M1

    def do(self) -> None:
        self.find_ellipse()
        self.find_bullseye()

        return self