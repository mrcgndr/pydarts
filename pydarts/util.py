import numpy as np
from shapely.geometry import MultiLineString


def rotate_anchor(theta=0, phi=0, gamma=0, x=0, y=0):
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = (np.deg2rad(a) for a in [theta, phi, gamma])
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        #d = np.sqrt(height**2 + width**2)
        #f = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -x],
                        [0, 1, -y],
                        [0, 0, 1],
                        [0, 0, 1]])
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(rtheta), -np.sin(rtheta), 0],
                        [0, np.sin(rtheta), np.cos(rtheta), 0],
                        [0, 0, 0, 1]])
        RY = np.array([ [np.cos(rphi), 0, -np.sin(rphi), 0],
                        [0, 1, 0, 0],
                        [np.sin(rphi), 0, np.cos(rphi), 0],
                        [0, 0, 0, 1]])
        RZ = np.array([ [np.cos(rgamma), -np.sin(rgamma), 0, 0],
                        [np.sin(rgamma), np.cos(rgamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        # Composed rotation matrix with (RX, RY, RZ)
        R = RX@RY@RZ
        # Projection 3D -> 2D matrix
        A2 = np.array([ [1, 0, x, 0],
                        [0, 1, y, 0],
                        [0, 0, 1, 0]])
        # Final transformation matrix
        return A2@R@A1

def find_nearest(array: np.array, values: np.array) -> np.array:
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

def transform_lines(lines: MultiLineString, M: np.ndarray) -> MultiLineString:
    
    
    return

def get_rad(theta, phi, gamma):
    return (np.deg2rad(theta),
            np.deg2rad(phi),
            np.deg2rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (np.rad2deg(rtheta),
            np.rad2deg(rphi),
            np.rad2deg(rgamma))
