# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:31:44 2017

@author: sakurai
"""

from xml.etree import ElementTree

import numpy as np
import matplotlib.pyplot as plt
import cv2


class Camera(object):
    def __init__(self, filename_intrinsic, filename_extrinsic):
        self.A, self.dist_coef = self.load_intrinsic(filename_intrinsic)
        self.R, self.t = self.load_extrinsic(filename_extrinsic)
        self.A_inv = np.linalg.inv(self.A)
        self.R_inv = np.linalg.inv(self.R)

    def load_intrinsic(self, filename):
        root = ElementTree.parse(filename).getroot()
        intrinsic_matrix = np.array(' '.join(
            [s.strip() for s in root[0][3].text.strip().split('\n')]
            ).split(' '), dtype=np.float).reshape(3, 3)

        dist_coef = np.array(' '.join(
            [s.strip() for s in root[1][3].text.strip().split('\n')]
            ).split(' '), dtype=np.float)
        return intrinsic_matrix, dist_coef

    def load_extrinsic(self, filename):
        root = ElementTree.parse(filename).getroot()
        rotation_matrix = np.array([row.strip().split(' ') for row in
                                    root[0][3].text.strip().split('\n')],
                                   dtype=np.float)
        translation_vector = np.array(
            root[1][3].text.strip().split(' '), dtype=np.float).reshape(3, 1)
        return (rotation_matrix, translation_vector)

    def map_world_to_image(self, p_world):
        p_world = p_world.reshape(3, 1)
        p_image = self.A.dot(self.R.dot(p_world) + self.t)
        uv = p_image[:2] / p_image[2]
        return uv.ravel()

    def map_image_to_world(self, p_image):
        p_image = np.asarray(p_image)
        if p_image.size == 2:
            p_image = np.concatenate((p_image, [1]))
        p_image = p_image.reshape(3, 1)
        xyz = self.R_inv.dot(self.A_inv.dot(p_image) - self.t)
        xyz = xyz / xyz[2]

        def xy1_z(z):
            return xyz * z
        return xy1_z

    def undistort_image(self, image):
        return cv2.undistort(image, self.A, self.dist_coef)


if __name__ == '__main__':
    intrinsic_filename = 'camera1/intrinsic.xml'
    extrinsic_filename = 'camera1/extrinsic.xml'
    image_filename = 'camera1/view.bmp'

    camera = Camera(intrinsic_filename, extrinsic_filename)

    points = np.array([[0, 0, 0],
                       [1000, 0, 0],
                       [1000, 1000, 0],
                       [0, 1000, 0],
                       [0, 0, 1000],
                       [1000, 0, 1000],
                       [1000, 1000, 1000],
                       [0, 1000, 1000]], dtype=np.float)

    o_world = np.array([0, 0, 0], dtype=np.float)
    o_image = camera.map_world_to_image(o_world)
    print o_image

    image = plt.imread(image_filename)
    image = camera.undistort_image(image)

    ps_image = []
    for p_world in points:
        p_image = camera.map_world_to_image(p_world)
        ps_image.append(p_image)
    ps_image = np.vstack(ps_image)

    plt.imshow(image)
    plt.plot(ps_image.T[0], ps_image.T[1], '+r', markersize=15)
    plt.show()

#    # not implemented yet
#    xyz = camera.map_image_to_world([303, 179])
#    print xyz(0)
