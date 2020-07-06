import cv2
import numpy as np
import math
import os
from objloader1 import *
import sys
from collections import namedtuple
from OpenGL.GL import *
from OpenGL.GLU import *
import sys, pygame
from pygame.locals import *
from pygame.constants import *








def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = img.shape[:-1]

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))







class PoseEstimator(object):
    def __init__(self):
        self.H=[]
        # Use locality sensitive hashing algorithm
        flann_params = dict(algorithm = 6, table_number = 6,
                key_size = 12, multi_probe_level = 1)

        self.min_matches = 10
        self.cur_target = namedtuple('Current', 'image, rect, keypoints, descriptors, data')
        self.tracked_target = namedtuple('Tracked', 'target, points_prev, points_cur, H, quad')
        self.matches=None
        self.feature_detector = cv2.ORB_create(nfeatures=3000)
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.tracking_targets = []

    # Function to add a new target for tracking
    def add_target(self, image, rect, data=None):
        x_start, y_start, x_end, y_end = rect
        keypoints, descriptors = [], []
        for keypoint, descriptor in zip(*self.detect_features(image)):
            x, y = keypoint.pt
            if x_start <= x <= x_end and y_start <= y <= y_end:
                keypoints.append(keypoint)
                descriptors.append(descriptor)

        descriptors = np.array(descriptors, dtype='uint8')
        self.feature_matcher.add([descriptors])
        target = self.cur_target(image=image, rect=rect, keypoints=keypoints,
                    descriptors=descriptors, data=None)
        self.tracking_targets.append(target)

    # To get a list of detected objects
    def track_target(self, frame):
        
        self.cur_keypoints, self.cur_descriptors = self.detect_features(frame)
        if len(self.cur_keypoints) < self.min_matches:
            return []

        self.matches = self.feature_matcher.knnMatch(self.cur_descriptors, k=2)
        self.matches = [match[0] for match in self.matches if len(match) == 2 and
                    match[0].distance < match[1].distance * 0.75]
        if len(self.matches) < self.min_matches:
            return []

        matches_using_index = [[] for _ in range(len(self.tracking_targets))]
        for match in self.matches:
            matches_using_index[match.imgIdx].append(match)

        tracked = []
        # imgds=False
        # if not imgds:
        #     tracked.append(cv2.imread('book.jpg'))
        #     imgds=True
        for image_index, self.matches in enumerate(matches_using_index):
            if len(self.matches) < self.min_matches:
                continue

            target = self.tracking_targets[image_index]
            points_prev = [target.keypoints[m.trainIdx].pt for m in self.matches]
            points_cur = [self.cur_keypoints[m.queryIdx].pt for m in self.matches]
            points_prev, points_cur = np.float32((points_prev, points_cur))
            self.H, status = cv2.findHomography(points_prev, points_cur, cv2.RANSAC, 10.0)
         
            status = status.ravel() != 0
            if status.sum() < self.min_matches:
                continue

            points_prev, points_cur = points_prev[status], points_cur[status]

            x_start, y_start, x_end, y_end = target.rect
            quad = np.float32([[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), self.H).reshape(-1, 2)

            track = self.tracked_target(target=target, points_prev=points_prev,
                        points_cur=points_cur, H=self.H, quad=quad)
            tracked.append(track)
            
            
        tracked.sort(key = lambda x: len(x.points_prev), reverse=True)
        return tracked

    # Detect features in the selected ROIs and return the keypoints and descriptors
    def detect_features(self, frame):
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []

        return keypoints, descriptors

    # Function to clear all the existing targets
    def clear_targets(self):
        self.feature_matcher.clear()
        self.tracking_targets = []

class VideoHandler(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.paused = False
        self.frame = None
        self.pose_tracker = PoseEstimator()
        self.alpha = 0.4
        self.holoframe=None
        self.H=None
        cv2.namedWindow('Tracker')
        self.roi_selector = ROISelector('Tracker', self.on_rect)

    def on_rect(self, rect):
        self.pose_tracker.add_target(self.frame, rect)

    def start(self):
        
        while True:
            is_running = not self.paused and self.roi_selector.selected_rect is None

            if is_running or self.frame is None:
                ret, frame = self.cap.read()
               
                if not ret:
                    break
        
                self.frame = frame.copy()
            
            img = self.frame.copy()
            if is_running:
                tracked = self.pose_tracker.track_target(self.frame)
                
                for item in tracked:
                    self.frame=cv2.polylines(img, [np.int32(item.quad)],True, (255, 255, 0), 2)
                    h, w = img.shape[:-1]
                    K = np.float64([[w, 0, 0.1*(w-1)],
                    [-1, w, 0.1*(h-1)],
                    [-1, -1, -1.0]])
                    projection = projection_matrix(camera_parameters, self.pose_tracker.H)  
                            # project cube or model
                    self.frame = render(self.frame, obj, projection, item, False)
                    # self.frame = cv2.drawMatches(img, self.pose_tracker.cur_descriptors, frame, self.pose_tracker.cur_keypoints, self.pose_tracker.matches[:10], 0, flags=2)
            
                            # frame = render(frame, model, projection)
                    
                    
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(img, (x, y), 2, (255, 255, 255))

            self.roi_selector.draw_rect(img)
            cv2.imshow('Tracker', img)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.pose_tracker.clear_targets()
            if ch == 27:
                break

class ROISelector(object):
    def __init__(self, win_name, callback_func):
        self.win_name = win_name
        self.callback_func = callback_func
        cv2.setMouseCallback(self.win_name, self.on_mouse_event)
        self.selection_start = None
        self.selected_rect = None

    def on_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection_start = (x, y)

        if self.selection_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                x_orig, y_orig = self.selection_start
                x_start, y_start = np.minimum([x_orig, y_orig], [x, y])
                x_end, y_end = np.maximum([x_orig, y_orig], [x, y])
                self.selected_rect = None
                if x_end > x_start and y_end > y_start:
                    self.selected_rect = (x_start, y_start, x_end, y_end)
            else:
                rect = self.selected_rect
                self.selection_start = None
                self.selected_rect = None
                if rect:
                    self.callback_func(rect)

    def draw_rect(self, img):
        if not self.selected_rect:
            return False

        x_start, y_start, x_end, y_end = self.selected_rect
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        return True

if __name__ == '__main__':
    
    
    # Compute model keypoints and its descriptors
    # Load 3D model from OBJ file
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    
    # pygame.init()
    # viewport = (800,600)
    # hx = viewport[0]/2
    # hy = viewport[1]/2
    # srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

    # glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
    # glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
    # glEnable(GL_LIGHT0)
    # glEnable(GL_LIGHTING)
    # glEnable(GL_COLOR_MATERIAL)
    # glEnable(GL_DEPTH_TEST)
    # glShadeModel(GL_SMOOTH)   
    
    
    obj = OBJ('fox.obj', swapyz=True)
    imgds=False
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # width, height = viewport
    # gluPerspective(90.0, width/float(height), 1, 100.0)
    # glEnable(GL_DEPTH_TEST)
    # glMatrixMode(GL_MODELVIEW)
    VideoHandler().start()












