from operator import truth
from dataclasses import dataclass
import sys
import asyncio



sys.path.append(".")
# from pye3d.camera import CameraModel
# from pye3d.detector_3d import Detector3D, DetectorMode


import queue
import threading
import numpy as np
import cv2

from blink import *
from blob import *
from enum import Enum


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


class EyeProcessor:
    def __init__(self):
        self.calibration_frame_counter = None
        self.xmax = -69420
        self.xmin = 69420
        self.ymax = -69420
        self.ymin = 69420
        self.blink_clear = False
        self.blink_state = False
        self.min_int = 0

        self.camera_model = None
        self.detector_3d = None
        self.er_hsf = None
        self.er_hsrac = None
        self.er_daddy = None
        self.er_leap = None

    def BLINKM(self):
        self.eyeopen = BLINK(self, max_len=500)

    def LEAPM(self):
        self.thresh = self.current_image_gray.copy()
        (self.current_image_gray, self.rawx, self.rawy, self.eyeopen,) = self.er_leap.run(
            self.current_image_gray, self.current_image_gray_clean, self.calibration_frame_counter, self.settings.leap_calibration_samples
        )  # TODO: make own self var and LEAP toggle
        self.thresh = self.current_image_gray.copy()
        # todo: lorow, fix this as well
        self.out_x, self.out_y, self.avg_velocity = cal.cal_osc(self, self.rawx, self.rawy, self.angle)
        # self.current_algorithm = EyeInfoOrigin.LEAP

    def BLOB(self):
        self.blob = BLOB(self, max_len=500)


    def DETECTM(self):
        self.current_image_gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.current_image_gray_clean = (
                self.current_image_gray.copy()
            )  # copy this frame to have a clean image for blink algo
        self.BLINKM(self)
        if self.blink_state:
            return
        else: self.LEAPM()
    


    def UPDATE(self, current_image):
        self.current_image = current_image



if __name__ == "__main__":
    eye_processor = EyeProcessor()
    eye_processor.BLINKM()

 