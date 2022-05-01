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
import time


def show_cluster(px, shape):
    arr = np.zeros(shape)
    for i,j in px:
        arr[i,j] = 255
    plt.matshow(arr)
    plt.show()

def show_many_clusters(clusters, shape):
    arr = np.zeros(shape)
    for i, cluster in enumerate(clusters):
        for r,c in cluster:
            try:
                arr[r,c] = i+1
            except:
                pass
    plt.matshow(arr)
    plt.show()

def get_color_clusters(im, n_clusters=2):
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
    print(f"getting all_clusters[] took {tock-tick}")
    return all_clusters, clusters, kmap, num_clusters

def cluster_to_img(cluster, shape):
    arr = np.zeros(shape)
    for r, c in cluster:
        arr[r,c] = 255
    return arr.astype(np.uint8)

def is_cluster_in_bounds(cluster, rowbounds, colbounds):
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

def lineness_metric(cluster, n_buckets=30):
    # get the variance of frequencies of angles of pixels around the center
    # circles -> low variance, lines -> high variance
    cluster = np.array(cluster)  # just in case
    center = [np.mean(cluster[:,0]), np.mean(cluster[:,1])]
    de_meaned = cluster - center
    angles = np.array([np.arctan2(o, a) for o,a in de_meaned])
    angle_freqs = np.histogram(angles, bins=n_buckets, range=(-pi, pi))[0]
    scaled_angle_freqs = angle_freqs/len(angles)
    return np.std(scaled_angle_freqs), scaled_angle_freqs

def get_dial_cluster_angle(cluster, center):
    de_centered = cluster - center
    center_of_mass = [np.mean(de_centered[:,0]), np.mean(de_centered[:,1])]
    # coords are like
    # +---> y
    # | \
    # |  `' angle opens towards y
    # x
    angle = np.arctan2(center_of_mass[1], center_of_mass[0])
    # fix reference
    angle = (-angle + pi)
    return angle


def get_angle(im):
    # open image
    #tick = time.time()
    # img = Image.open(filepath)
    # width, height = img.size
    # resize_factor = 800/width
    # img_shape = (int(width * resize_factor), int(height * resize_factor))
    # img = img.resize(img_shape)
    # img_shape = img_shape[::-1] # we need this for np now
    # im = np.array(img)
    # im = im[:,:,:3] # just in case there's an alpha channel
    # print(f"Image shape (np) is {im.shape}")
    #tock = time.time()
    #print(f"opening + reshaping image took {tock-tick}") # opening + reshaping image took 0.7760021686553955
    img_shape = im.shape[:2]

    # find the circles
    #tick = time.time()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=rows / 8,
        param1=300, param2=100,
        minRadius=int(rows/8), maxRadius=500
    ) # TODO: check param1 and param2
    if circles is None:
        print("no circles?")
        return
    tock = time.time()
    #print(f"finding circles took {tock-tick}") # finding circles took 0.07358479499816895

    # get circle containing userpoint
    userpoint = [img_shape[0]/2, img_shape[1]/2] #[390, 270]
    # ^ TODO: get userpoint from UI
    circle = None
    for col, row, rad in circles[0]:
        if (userpoint[0]-col)**2 + (userpoint[1]-row)**2 < rad**2:
            circle = [col, row, rad]
    if not circle:
        print("userpoint not in circle?")
        return
    
    # get the bounds of the circle
    colbounds = [circle[0]-circle[2], circle[0]+circle[2]]
    rowbounds = [circle[1]-circle[2], circle[1]+circle[2]]
    circlecenter = [circle[1], circle[0]]  # this is in [row, col] order (numpy order)

    # get clusters of similarly colored pixles, and choose the ones inside the circle bounds
    tick = time.time()
    all_clusters, clusters, kmap, _ = get_color_clusters(im, n_clusters=5) # TODO: this one takes a while
    tock = time.time()
    print(f"getting all clusters took {tock-tick}")
    #tick = time.time()
    filtered_clusters = [c for c in all_clusters if len(c) > 100 and is_cluster_in_bounds(c, rowbounds, colbounds)]
    # TODO: ^ get a better cluster length bound
    print(f"{len(filtered_clusters)} of {len(all_clusters)} are in the bounds and large")
    #tock = time.time()
    #print(f"filtering clusters took {tock-tick}") # filtering clusters took 0.3705437183380127

    show_many_clusters(filtered_clusters, img_shape)

    #tick = time.time()
    # get the lines of the clusters
    all_lines = []
    line_clusters = []
    for i, cluster in enumerate(filtered_clusters):
        lines = cv2.HoughLines(cluster_to_img(cluster, img_shape), 1, 0.01, int(3000/np.sqrt(len(cluster))))
        if lines is not None:
            all_lines.append(lines[0])
            line_clusters.append(cluster)
    print(f"{len(all_lines)} clusters gave out lines")
    #tock = time.time()
    #print(f"getting lines took {tock-tick}") # getting lines took 0.4962620735168457

    # get linenesses
    #tick = time.time()
    linenesses = [lineness_metric(cluster)[0] for cluster in line_clusters]
    #tock = time.time()
    #print(f"getting linenesses took {tock-tick}") # getting linenesses took 0.2713451385498047

    # get cluster that's most like a line
    line_cluster = line_clusters[np.argmax(linenesses)]
    show_cluster(line_cluster, img_shape)
    print(f"the line's cluster is {line_cluster.shape} pixels")

    # get its angle
    angle = get_dial_cluster_angle(line_cluster, circlecenter)
    return angle


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("filepath?")
        exit()
    filepath = sys.argv[1]

    # open image
    #tick = time.time()
    img = Image.open(filepath)
    width, height = img.size
    resize_factor = 800/width
    img_shape = (int(width * resize_factor), int(height * resize_factor))
    img = img.resize(img_shape)
    img_shape = img_shape[::-1] # we need this for np now
    im = np.array(img)
    im = im[:,:,:3] # just in case there's an alpha channel
    print(f"Image shape (np) is {im.shape}")
    #tock = time.time()
    #print(f"opening + reshaping image took {tock-tick}") # opening + reshaping image took 0.7760021686553955

    angle = get_angle(im)
    print(f"Angle is {angle} radians from vertical, increasing clockwise")
