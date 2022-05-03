import sys
import cv2
from PIL import Image
import numpy as np
import math
from math import pi
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.measure import label
from sklearn.feature_extraction import image
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.transform import AffineTransform, warp
from skimage.feature import canny
from skimage import color
import time
from time import sleep
import json
import threading
import signal

import RPi.GPIO as GPIO
#import berryIMU
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
ledPin = ledPin1 = 22
ledPin2 = 12
GPIO.setup(ledPin1, GPIO.OUT)
GPIO.setup(ledPin2, GPIO.OUT)


class Configerator():
    def __init__(self, debug=False):
        self.debug = debug
        self.proposed_circles = None
        self.config = dict()
    
    def open_cameras(self):
        self.cam1 = cv2.VideoCapture(0)
        self.cam2 = cv2.VideoCapture(2)
        self.cam1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cam2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cam1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cam2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        GPIO.output(ledPin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(ledPin, GPIO.LOW) 
        time.sleep(0.5)
        print("configeragor opened cameras")
        
    def close_cameras(self):
        self.cam1.release()
        self.cam2.release()
        print("configerator closed cameras")
        GPIO.output(ledPin1, GPIO.LOW)
        GPIO.output(ledPin2, GPIO.LOW)

    def getImage(self):
        self.cam1.grab()
        ret1, image1 = self.cam1.retrieve()
        if not ret1:
            raise Exception("configerator couldn't get camera's image!")
        return image1  # numpy

    def getProposedCircles(self):
        im = self.getImage()
        self.current_image = im.copy()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=rows / 8,
            param1=300, param2=100,
            minRadius=int(rows/8), maxRadius=5000
        ) # TODO: check param1 and param2

        if self.debug:
            # if this is a debugging case:
            # make sure there are circles
            if circles is None:
                circles = [[[50, 50, 50], [150, 150, 100]]] # col, row, radius

        if circles is not None:
            circles = np.uint16(np.around(circles))[0]
            self.proposed_circles = circles
            for i in circles:
                center = (i[0], i[1])
                # circle center
                #cv2.circle(im, center, 5, (0, 255, 255), 3)
                # circle outline
                radius = i[2]
                cv2.circle(im, center, radius, (255, 0, 255), 3)

        
        img = Image.fromarray(im)
        img.save('config_circles.png')
        return len(circles) if circles is not None else 0 # TODO this is wrong for testing

    def sendUserpoint(self, circle_num):
        # get circle containing userpoint
        if self.proposed_circles is None:
            raise Exception("can't compare userpoint against nonexistent proposed circles!")
        #for col, row, rad in self.proposed_circles:
        #    if (userpoint[0]-col)**2 + (userpoint[1]-row)**2 < rad**2:
        #        self.circle = (col, row, rad)  # cv2
        #        self.config["userpoint"] = (row, col)  # numpy order
        #        self.config["circle"] = (row, col, rad)
        # STORE EVERYTHING AS NUMPY-ORDER
        self.circle = self.proposed_circles[circle_num]
        self.config["userpoint"] = [
            int(self.circle[0]),
            int(self.circle[1])]
        self.config["circle"] = [
            int(self.circle[0]),
            int(self.circle[1]),
            int(self.circle[2])
        ] # bah
 
    def getLineImage(self, lineNum):
        #im = self.getImage()
        im = self.current_image.copy()
        pt1 = self.circle[:2]  # (col, row)
        if lineNum==1:  # horizontal going right
            pt2 = (pt1[0] + 1000, pt1[1])
        elif lineNum==2:  # going down
            pt2 = (pt1[0], pt1[1] + 1000)
        cv2.line(im, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        img = Image.fromarray(im)
        img.save(f'config_line{lineNum}.png')
        return im

    def setLine(self, userInputs):
        assert len(userInputs) == 2
        # userInputs is a list of (reading1, reading2)
        # which correspond to right line and down line
        # TODO input validation
        speed_over_angle = (int(userInputs[1]) - int(userInputs[0])) / (pi/2)
        self.config["speed_over_angle"] = float(speed_over_angle)

    def writeConfig(self, config_path):
        if "userpoint" not in self.config \
          or "speed_over_angle" not in self.config \
          or "circle" not in self.config:
            raise Exception("can't write config file without config values! self.config is", self.config)
        if self.debug:
            print("config is", self.config)
        with open(config_path, "w+") as f:
            json.dump(self.config, f)


class SpeedFinder():
    def __init__(self, config_path="config.json", debug=True):
        self.reset_config(config_path)
        self.default_config(speed_over_angle=90/(pi/2), tach_speed_over_angle=15/(pi/2))
        self.debug = debug
        self.exit_event = None
        self.is_running = False
        # for dial reading
        self.airspeed_circ = None
        self.tach_circ = None
        # for cable tracking: default values
        self.cable_middle_r = 597.6418
        self.cable_range_r = 89.352

    def reset_config(self, config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        try:
            self.userpoint = config_dict["userpoint"]
            self.speed_over_angle = config_dict["speed_over_angle"]
            self.circle = config_dict["circle"]
        except Exception as e:
            raise Exception("error reading config dict", e)

    def default_config(self, speed_over_angle, tach_speed_over_angle):
        self.speed_over_angle = speed_over_angle
        self.tach_speed_over_angle = tach_speed_over_angle

    #def reset_config(self, userpoint, speed_over_angle, circle):
    #    self.userpoint = userpoint
    #    self.speed_over_angle = speed_over_angle
    #    self.circle = circle  # (row, col, rad)
    #    self.airspeed_circ = None
    #    self.tach_circ = None

    def unwarp_image(self,im, t_inv):
        t_inv /= np.sqrt(np.linalg.det(t_inv)) / 1.5  # make it a bit bigger
        t_inv_full = np.array([[t_inv[0,0], t_inv[0,1], 0.],
                            [t_inv[1,0], t_inv[1,1], 0.],
                            [0.0,        0.0,        1.]])
        t = np.linalg.inv(t_inv_full)
        affine_tf = AffineTransform(matrix=t)
        unwarped = warp(im, affine_tf)
        return unwarped

    def get_dial_ijr(self, im_unwarped):
        gray = color.rgb2gray(im_unwarped)
        edges = canny(gray, sigma=0.7, low_threshold=0.55, high_threshold=0.9)
        hough_radii = np.arange(50, 200, 1)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, #min_xdistance=20, min_ydistance=20,
                                            total_num_peaks=1)
        circles = list(zip(cy, cx, radii))
        return circles[0]
    
    def shrink_im_to_circle(self, im, i, j, r):
        rowbounds = [i-r, i+r]
        colbounds = [j-r, j+r]
        smaller_im = im[rowbounds[0]:rowbounds[1], colbounds[0]:colbounds[1]]
        return smaller_im

    def show_cluster(self, px, shape):
        if threading.current_thread() != threading.main_thread():
            return
        arr = np.zeros(shape)
        for i,j in px:
            arr[i,j] = 255
        plt.matshow(arr)
        plt.show()

    def show_many_clusters(self, clusters, shape):
        if threading.current_thread() != threading.main_thread():
            return
        arr = np.zeros(shape)
        for i, cluster in enumerate(clusters):
            for r,c in cluster:
                try:
                    arr[r,c] = i+1
                except:
                    pass
        plt.matshow(arr)
        plt.show()

    def get_color_clusters(self, im, n_clusters=2):
        # im: numpy array of a 3-channel image
        colors = im.reshape(-1,3)  # a list of all the pixel colors in the image
        #tick = time.time()
        kmeans = KMeans(n_clusters=n_clusters).fit(colors)
        #tock = time.time()
        #print(f"kmeans took {tock-tick}") # kmeans took 1.6302671432495117
        # ^ KMeans() is from sklearn.cluster
        kmap = kmeans.labels_.copy().reshape(im.shape[0:2])  # an image where each pixel is replaced by its kmeans label
        #tick = time.time()
        clusters, num_clusters = label(kmap, connectivity=1, background=-1, return_num=True)  # an image where each pixel is replaced by its cluster label
        #tock = time.time()
        #print(f"label() took {tock-tick}") # label() took 0.013054370880126953
        # ^ label() is from skimage.measure
        tick = time.time()
        all_clusters = [np.transpose(np.where(clusters==n)) for n in range(num_clusters)]  # a list of the coordinates of pixels in each cluster
        # ^ TODO: this line is taking a while
        tock = time.time()
        #print(f"getting all_clusters[] took {tock-tick}")
        return all_clusters, clusters, kmap, num_clusters

    def cluster_to_img(self, cluster, shape):
        arr = np.zeros(shape)
        for r, c in cluster:
            arr[r,c] = 255
        return arr.astype(np.uint8)

    def is_cluster_in_bounds(self, cluster, rowbounds, colbounds):
        try:
            cluster_rowmin = np.min(cluster[:,0])
            cluster_rowmax = np.max(cluster[:,0])
            cluster_colmin = np.min(cluster[:,1])
            cluster_colmax = np.max(cluster[:,1])
            cluster_center = [np.mean(cluster[:,0]), np.mean(cluster[:,1])]
            if cluster_rowmin > rowbounds[0] and cluster_rowmax < rowbounds[1] \
            and cluster_colmin > colbounds[0] and cluster_colmax < colbounds[1]:
                return True
            return False
        except:
            return False
    
    def is_cluster_in_circle(self, cluster, i, j, r):
        cluster_centered = cluster - [i,j]
        px_dists_sq = np.sum(np.power(cluster_centered, 2), axis=1)
        return np.max(px_dists_sq) <= (r+5)*(r+5) # leave a bit of buffer

    def lineness_metric(self, cluster, center=None, n_buckets=30):
        # get the variance of frequencies of angles of pixels around the center
        # circles -> low variance, lines -> high variance
        cluster = np.array(cluster)  # just in case
        if center is None:
            center = [np.mean(cluster[:,0]), np.mean(cluster[:,1])]
            using_center_of_mass = True
        else:
            using_center_of_mass = False
        de_meaned = cluster - center
        angles = np.array([np.arctan2(o, a) for o,a in de_meaned])
        if using_center_of_mass:
            # A dial needle extends in 2 directions away from its CoM,
            # so deduplicate that by checking angles in range (-pi/2, pi/2)
            angle_freqs = np.histogram(angles, bins=n_buckets, range=(-pi/2, pi/2))[0]
        else:
            # Then we are checking angles around the center of the dial;
            # all pixels of the dial needle should be at one angle
            angle_freqs = np.histogram(angles, bins=n_buckets, range=(-pi, pi))[0]
        scaled_angle_freqs = angle_freqs/len(angles)
        return np.std(scaled_angle_freqs), scaled_angle_freqs

    def get_dial_cluster_angle(self, cluster, center):
        de_centered = cluster - center
        center_of_mass = [np.mean(de_centered[:,0]), np.mean(de_centered[:,1])]
        # coords are like
        # +---> y
        # |u\
        # |  \  angle opens towards y
        # x   `
        angle = np.arctan2(center_of_mass[1], center_of_mass[0])
        # fix reference
        angle = (-angle + pi)
        return angle
    
    def calibrate_cable(self):
        # FUTURE WORK: calibrate based on the user's own video
        locs = []
        for i in range(1,271):
            img = Image.open(f"vid1/video-frame{i:05}.png") # next image of cable
            im = np.array(img)
            loc = self.get_cable_loc(im)
            locs.append(loc)
        locs = np.array(locs)
        
        lower_r = np.min(locs[:,0])
        upper_r = np.max(locs[:,0])
        middle_r = (upper_r + lower_r)/2
        range_r = upper_r - lower_r
        
        self.middle_r = middle_r
        self.range_r = range_r
        return middle_r, range_r

    def get_cable_loc(self, im):
        # This function expects a very specific image
        # FUTURE WORK: allow the user to specify cable position in the camera
        cable_patch = im[550:720, 900:1000]
        indicator_img = (cable_patch<[20,20,20]).astype(int).sum(axis=2)
        indicator_px = np.transpose(np.where(indicator_img>0))
        rel_loc = [np.mean(indicator_px[:,0]), np.mean(indicator_px[:,1])]
        abs_loc = [rel_loc[0] + 550, rel_loc[1]+900]
        return abs_loc
    
    def get_cable_pos(self, im):
        abs_r = self.get_cable_loc(im)[0]
        scaled_r = (abs_r - self.middle_r)/self.range_r*2
        return scaled_r
    
    def get_airspeed_and_tach(self, im):
        # This function expects a very specific image
        # FUTURE WORK: detect positions and view angles of dials automatically
        # Get the parts of the image with the dials
        airspeed_im = im[0:350,500:800]
        tach_im = im[200:500,300:500]

        # Unwarp the dials (the image was taken at an angle)
        t_inv_airspeed = np.array(
            [[0.65, -.25], 
             [0.1,   0.63]])
        airspeed_unwarped = self.unwarp_image(airspeed_im, t_inv_airspeed)
        t_inv_tach = np.array(
            [[0.68, -.32], 
             [0.1,   0.6]])
        tach_unwarped = self.unwarp_image(tach_im, t_inv_tach)
        if self.debug:
            plt.imshow(airspeed_unwarped)
            plt.imshow(tach_unwarped)
        
        # Find the circles of the dials: (row, col, radius)
        # also cache the results
        if self.airspeed_circ is None:
            airspeed_circ = self.get_dial_ijr(airspeed_unwarped)
            self.airspeed_circ = airspeed_circ
        else:
            airspeed_circ = self.airspeed_circ
        if self.tach_circ is None:
            tach_circ = self.get_dial_ijr(tach_unwarped)
            self.tach_circ = tach_circ
        else:
            tach_circ = self.tach_circ

        # Crop the images to the circles
        airspeed_small = self.shrink_im_to_circle(airspeed_unwarped, *airspeed_circ)
        airspeed_shape = airspeed_small.shape[:2]
        airspeed_center = (int(airspeed_shape[0]/2), int(airspeed_shape[1]/2))
        tach_small = self.shrink_im_to_circle(tach_unwarped, *tach_circ)
        tach_shape = tach_small.shape[:2]
        tach_center = (int(tach_shape[0]/2), int(tach_shape[1]/2))

        # Read the dials
        def max_lineness(cluster, smaller_im_center):
            # sometimes the center of the detected circle isn't the center of the dial :(
            a = self.lineness_metric(cluster, smaller_im_center)[0]
            b = self.lineness_metric(cluster)[0]
            return max(a,b)

        # AIRSPEED
        try:
            all_airspeed_clusters, clusters, kmap, _ = self.get_color_clusters(airspeed_small, n_clusters=2) # TODO: this one takes a while
        except:
            print("error getting clusters from smaller airspeed image")
            print(f"airspeed_shape={airspeed_shape}")
            return None
        filtered_airspeed_clusters = [c for c in all_airspeed_clusters if len(c) > 100 and self.is_cluster_in_circle(c, *airspeed_center, airspeed_circ[2])]
        if self.debug:
            print(f"{len(filtered_airspeed_clusters)} of {len(all_airspeed_clusters)} airspeed clusters are in the bounds and large")
        self.show_many_clusters(filtered_airspeed_clusters, airspeed_shape)
        all_airspeed_lines = []
        line_airspeed_clusters = []
        for i, cluster in enumerate(filtered_airspeed_clusters):
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False) #new!
            h, theta, d = hough_line(self.cluster_to_img(cluster, airspeed_shape), theta=tested_angles) #improved!
            lines = list(zip(*hough_line_peaks(h, theta, d)))
            if lines[0][0] > 30:
                all_airspeed_lines.append(lines)
                line_airspeed_clusters.append(cluster)
        if self.debug:
            print(f"{len(all_airspeed_lines)} airspeed clusters gave out lines")
        linenesses_airspeed = [max_lineness(cluster, airspeed_center) for cluster in line_airspeed_clusters]
        line_airspeed_cluster = line_airspeed_clusters[np.argmax(linenesses_airspeed)]
        self.show_cluster(line_airspeed_cluster, airspeed_shape)
        airspeed_angle = self.get_dial_cluster_angle(line_airspeed_cluster, airspeed_center)
        airspeed = airspeed_angle*self.speed_over_angle

        # TACH
        try:
            all_tach_clusters, clusters, kmap, _ = self.get_color_clusters(tach_small, n_clusters=4) # TODO: this one takes a while
        except:
            print("error getting clusters from smaller airspeed image")
            print(f"tach_shape={tach_shape}")
            return None
        filtered_tach_clusters = [c for c in all_tach_clusters if len(c) > 100 and self.is_cluster_in_circle(c, *tach_center, tach_circ[2])]
        if self.debug:
            print(f"{len(filtered_tach_clusters)} of {len(all_tach_clusters)} tach clusters are in the bounds and large")
        self.show_many_clusters(filtered_tach_clusters, tach_shape)
        all_tach_lines = []
        line_tach_clusters = []
        for i, cluster in enumerate(filtered_tach_clusters):
            tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False) #new!
            h, theta, d = hough_line(self.cluster_to_img(cluster, tach_shape), theta=tested_angles) #improved!
            lines = list(zip(*hough_line_peaks(h, theta, d)))
            if lines[0][0] > 30:
                all_tach_lines.append(lines)
                line_tach_clusters.append(cluster)
        if self.debug:
            print(f"{len(all_tach_lines)} tach clusters gave out lines")
        linenesses_tach = [max_lineness(cluster, tach_center) for cluster in line_tach_clusters]
        line_tach_cluster = line_tach_clusters[np.argmax(linenesses_tach)]
        self.show_cluster(line_tach_cluster, tach_shape)
        tach_angle = self.get_dial_cluster_angle(line_tach_cluster, tach_center)
        tach = tach_angle*self.tach_speed_over_angle  # TODO: specify the position of 0 on the dial

        return airspeed, tach
    
    def get_speed(self, im):
        # This function is for the demo
        img_shape = im.shape[:2]

        userpoint = self.userpoint
        circle = self.circle  # (row, col, rad)
        
        # get the bounds of the circle
        rowbounds = [circle[0]-circle[2], circle[0]+circle[2]]
        colbounds = [circle[1]-circle[2], circle[1]+circle[2]]
        circlecenter = circle[:2]  # this is in [row, col] order (numpy order)

        # get clusters of similarly colored pixles, and choose the ones inside the circle bounds
        tick = time.time()
        smaller_im = im[rowbounds[0]:rowbounds[1], colbounds[0]:colbounds[1]]
        smaller_im_shape = (rowbounds[1]-rowbounds[0], colbounds[1]-colbounds[0])
        smaller_im_center = (int(smaller_im_shape[0]/2), int(smaller_im_shape[1]/2))
        try:
            all_clusters, clusters, kmap, _ = self.get_color_clusters(smaller_im, n_clusters=5) # TODO: this one takes a while
        except:
            print("error getting clusters from smaller image")
            print(f"smaller_im_shape={smaller_im_shape}, circle={circle}, rowbounds={rowbounds}, colbounds={colbounds}")
            return None
        tock = time.time()
        #print(f"getting all clusters took {tock-tick}")
        #tick = time.time()
        filtered_clusters = [c for c in all_clusters if len(c) > 100 and self.is_cluster_in_circle(c, *smaller_im_center, circle[2])]
        #filtered_clusters = [c for c in all_clusters if len(c) > 100]
        # TODO: ^ get a better cluster length bound
        if self.debug:
            print(f"{len(filtered_clusters)} of {len(all_clusters)} are in the bounds and large")
        #tock = time.time()
        #print(f"filtering clusters took {tock-tick}") # filtering clusters took 0.3705437183380127

        self.show_many_clusters(filtered_clusters, smaller_im_shape)

        #tick = time.time()
        # get the lines of the clusters
        all_lines = []
        line_clusters = []
        for i, cluster in enumerate(filtered_clusters):
            lines = cv2.HoughLines(self.cluster_to_img(cluster, smaller_im_shape), 1, 0.01, int(3000/np.sqrt(len(cluster))))
            if lines is not None:
                all_lines.append(lines[0])
                line_clusters.append(cluster)
        if self.debug:
            print(f"{len(all_lines)} clusters gave out lines")
        #tock = time.time()
        #print(f"getting lines took {tock-tick}") # getting lines took 0.4962620735168457

        # get linenesses
        #tick = time.time()
        if not line_clusters:
            print("no lines found")
            return None
        
        def max_lineness(cluster, smaller_im_center):
            # sometimes the center of the detected circle isn't the center of the dial :(
            a = self.lineness_metric(cluster, smaller_im_center)[0]
            b = self.lineness_metric(cluster)[0]
            return max(a,b)
        
        #linenesses = [self.lineness_metric(cluster)[0] for cluster in line_clusters]
        linenesses = [max_lineness(cluster, smaller_im_center) for cluster in line_clusters]
        #tock = time.time()
        #print(f"getting linenesses took {tock-tick}") # getting linenesses took 0.2713451385498047

        # get cluster that's most like a line
        line_cluster = line_clusters[np.argmax(linenesses)]
        
        self.show_cluster(line_cluster, smaller_im_shape)
        #print(f"the line's cluster is {line_cluster.shape} pixels")

        # get its angle
        angle = self.get_dial_cluster_angle(line_cluster, (smaller_im_shape[0]/2, smaller_im_shape[1]/2))
        return angle*self.speed_over_angle
    
    def _run(self, exit_event):
        data_path = f"/home/pi/Desktop/Senior_Design_Project/camera_data_{time.time()}.txt"
        f = open(data_path, "w+")
        #cam1 = cv2.VideoCapture(0)
        #cam2 = cv2.VideoCapture(2)
        cam1 = cv2.VideoCapture("/dev/video0")
        cam2 = cv2.VideoCapture("/dev/video2")
        cam1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cam2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        
        # get rid of buffer so we always grab the latest frame
        cam1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_width1 = int(cam1.get(3))
        frame_height1 = int(cam1.get(4))
        frame_width2 = int(cam2.get(3))
        frame_height2 = int(cam2.get(4))

        frame_size1 = (frame_width1, frame_height1)
        frame_size2 = (frame_width2, frame_height2)
        # writer1 = cv2.VideoWriter('/home/pi/Desktop/Senior_Design_Project/test1_analysis.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size1)
        # writer2 = cv2.VideoWriter('/home/pi/Desktop/Senior_Design_Project/test2_analysis.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size2)
        while True:
            try:
                GPIO.output(ledPin, GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(ledPin, GPIO.LOW) 
                #time.sleep(0.5)
                if exit_event is not None and exit_event.is_set():
                    break
                tick = time.time()
                #ret1, image1 = cam1.read()
                cam1.grab()
                grabtime = time.time()
                ret1, image1 = cam1.retrieve()
                ret2, image2 = cam2.read()
                tock = time.time()
                #print(f"cam.read() took {tock-tick}")
                #hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
                #out.write(hsv)
                if ret1:
                    image_type = type(image1)
                    #print(f"image type: {type(image)}")
                    #print("image type", image_type)
                    if self.debug:
                        print("image shape", image1.shape)
                    #cv2.imshow('Web1', image1)
                    #cv2.imshow('Web2', image2)
                    if threading.current_thread()==threading.main_thread():
                        plt.imshow(image1)
                        plt.show()
                    speed = self.get_speed(image1)
                    if self.debug:
                        print(f"speed is {speed}")
                    if speed is not None:
                        f.write(f"{grabtime}, {speed}\n")
                        f.flush()
                else:
                    if self.debug:
                        print("no ret1 this time")
                    sleep(1)
                if ret2:
                    pass
            except KeyboardInterrupt:
                print(" kb interrupt")
                break
        cam1.release()
        cam2.release()
        # writer1.release()
        # writer2.release()
        cv2.destroyAllWindows()
        f.close()
        print("speedfinder camera stop")
        GPIO.output(ledPin1, GPIO.LOW)
        GPIO.output(ledPin2, GPIO.LOW)
    
    def start(self):
        # just in case we get a sigint or sigterm,
        # try to clean up properly
        def signal_handler(signum, frame):
            if self.exit_event is not None:
                self.exit_event.set()
                self.exit_event = None
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if self.exit_event is None:
            self.exit_event = threading.Event()
            self.t = threading.Thread(target=self._run, args=(self.exit_event,))
            self.t.start()
            print("speedfinder started")
            self.is_running = True
    
    def stop(self):
        if self.exit_event is not None:
            self.exit_event.set()
            self.t.join()
            print("speedfinder stopped")
            self.exit_event = None
            self.is_running = False
