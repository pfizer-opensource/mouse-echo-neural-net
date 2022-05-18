import numpy as np
import pandas as pd
import scipy
import cv2
import pydicom
from skimage.morphology import remove_small_holes
from skimage.morphology import closing, disk
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def postprocess(y_val_pred, apply_closing=True):
    """
    Fill holes in raw predications, followed by closing operation
    :param y_val_pred: segmentation output, batch x height x width x numLabels
    :return: out: batch x height x width
    """
    out = np.zeros_like(y_val_pred, dtype=np.uint8)

    masks_pred = np.argmax(y_val_pred, axis=3)
    for i in range(masks_pred.shape[0]):
        for j in range(4):
            temp = (masks_pred[i, :, :] == j)
            temp = remove_small_holes(temp, area_threshold=1280)
            out[i, :, :, j] = np.logical_not(remove_small_holes(np.logical_not(temp), area_threshold=1280))

    out = np.argmax(out, axis=3)

    if apply_closing:
        for i in range(out.shape[0]):
            out[i, :, :] = closing(out[i, :, :], disk(6))

        # after closing, fill holes again
        y_val_pred_temp = np.zeros_like(y_val_pred, dtype=np.uint8)
        for i in range(out.shape[0]):
            for j in range(4):
                temp = (out[i, :, :] == j)
                temp = remove_small_holes(temp, area_threshold=1280)
                y_val_pred_temp[i, :, :, j] = np.logical_not(remove_small_holes(np.logical_not(temp), area_threshold=1280))

        out = np.argmax(y_val_pred_temp, axis=3)

    return out


def plot_processing_image(image, label):
    """
    :param image:
    :param label:
    :return:
    """
    curve1 = []
    curve2 = []
    curve3 = []
    curve4 = []
    for col in label.T:
        ans = np.where(col == 1)
        curve1.append(ans[0][0])
        ans = np.where(col == 2)
        curve2.append(ans[0][0])
        ans = np.where(col == 3)
        curve3.append(ans[0][0])
        curve4.append(ans[0][-1])

    # Do detection based on difference between curve3 and curve2
    curve_dist = list(np.array(curve3) - np.array(curve2))
    plt.figure()
    plt.plot(curve_dist, 'r-')
    curve_dist = scipy.ndimage.gaussian_filter(curve_dist, 5)
    plt.plot(curve_dist, 'b-')

    # # Smooth all curves
    # curve1 = scipy.ndimage.gaussian_filter(curve1, 3)
    # curve2 = scipy.ndimage.gaussian_filter(curve2, 3)
    # curve3 = scipy.ndimage.gaussian_filter(curve3, 3)
    # curve4 = scipy.ndimage.gaussian_filter(curve4, 3)

    # Can probably increase distance to 50-100
    diastoles = scipy.signal.find_peaks(curve_dist, distance=30)
    diastoles = diastoles[0]
    systoles = scipy.signal.find_peaks(np.array(curve_dist) * -1, distance=30)
    systoles = systoles[0]

    # Generate Processing Figure
    img = cv2.merge((image, image, image))
    for diastole in diastoles:
        img = cv2.line(img, (diastole, curve1[diastole]), (diastole, curve4[diastole]), color=(255, 0, 0), thickness=2)
    for systole in systoles:
        img = cv2.line(img, (systole, curve1[systole]), (systole, curve4[systole]), color=(0, 0, 255), thickness=2)
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.plot(curve1, 'y--')
    plt.plot(curve2, 'y--')
    plt.plot(curve3, 'y--')
    for diastole in diastoles:
        plt.plot(diastole, curve3[diastole], 'r*', markersize=12)
    for systole in systoles:
        plt.plot(systole, curve3[systole], 'b*', markersize=12)
    plt.plot(curve4, 'y--')

    return


def getRes_rawDICOM(data):
    """
    Input: data (pydicom.dataset.FileDataset), read from raw DICOM

    Output: res_x (unit: mm)
            res_y (unit: mm)
    """

    res_y = float(data.PixelSpacing[0])
    res_x = float(data.PixelSpacing[1])

    return res_x, res_y


def compute_MMode_metrics(label, res_y, res_x, agg_fn=np.median):
    """
    :param label:
    :param res_y:
    :param res_x:
    :param agg_fn: function to aggregate cardiac cycles
    :return:
    """
    curve1 = []
    curve2 = []
    curve3 = []
    curve4 = []
    for col in label.T:
        # skip col where not all label = 0, 1, 2, 3 exist
        if len(np.unique(col)) != 4:
            curve1.append(0)
            curve2.append(0)
            curve3.append(0)
            curve4.append(0)
            continue

        ans = np.where(col == 1)
        curve1.append(ans[0][0])
        ans = np.where(col == 2)
        curve2.append(ans[0][0])
        ans = np.where(col == 3)
        curve3.append(ans[0][0])
        curve4.append(ans[0][-1])

    # Do detection based on difference between curve3 and curve2
    curve_dist = list(np.array(curve3) - np.array(curve2))
    curve_dist = scipy.ndimage.gaussian_filter(curve_dist, 5)

    # # Smooth all curves
    # curve1 = scipy.ndimage.gaussian_filter(curve1, 3)
    # curve2 = scipy.ndimage.gaussian_filter(curve2, 3)
    # curve3 = scipy.ndimage.gaussian_filter(curve3, 3)
    # curve4 = scipy.ndimage.gaussian_filter(curve4, 3)

    # Can probably increase distance to 50-100
    diastoles = scipy.signal.find_peaks(curve_dist, distance=15)
    diastoles = diastoles[0]
    systoles = scipy.signal.find_peaks(np.array(curve_dist) * -1, distance=15)
    systoles = systoles[0]

    # # Remove detected phases at beginning and end
    # if len(diastoles) > 4:
    #     print(diastoles)
    #     diastoles = sorted(diastoles)
    #     diastoles = diastoles[1:-1]
    # if len(systoles) > 4:
    #     print(systoles)
    #     systoles = sorted(systoles)
    #     systoles = systoles[1:-1]

    # compute metrics
    ### diastoles
    LVID_d_all = []
    LVAW_d_all = []
    LVPW_d_all = []
    for diastole in diastoles:
        LVAW_d_all.append(curve2[diastole] - curve1[diastole])
        LVID_d_all.append(curve3[diastole] - curve2[diastole])
        LVPW_d_all.append(curve4[diastole] - curve3[diastole])
    LVID_d = agg_fn(np.array(LVID_d_all)) * res_y
    LVAW_d = agg_fn(np.array(LVAW_d_all)) * res_y
    LVPW_d = agg_fn(np.array(LVPW_d_all)) * res_y

    ### systoles
    LVID_s_all = []
    LVAW_s_all = []
    LVPW_s_all = []
    for systole in systoles:
        LVAW_s_all.append(curve2[systole] - curve1[systole])
        LVID_s_all.append(curve3[systole] - curve2[systole])
        LVPW_s_all.append(curve4[systole] - curve3[systole])
    LVID_s = agg_fn(np.array(LVID_s_all)) * res_y
    LVAW_s = agg_fn(np.array(LVAW_s_all)) * res_y
    LVPW_s = agg_fn(np.array(LVPW_s_all)) * res_y

    ### Fractional Shortening
    FS = (LVID_d- LVID_s) / LVID_d * 100

    ### LV Mass
    LV_Mass = 1.053 * ((LVID_d + LVPW_d + LVAW_d)**3 - LVID_d**3)
    LV_Mass_Cor = 0.8 * LV_Mass

    ### Heart Rate; added March 8, 2021
    # BPM, 1 min has 60000 ms, res_x has ms unit
    Heart_Rate_sys = 60000.0 / (np.median(systoles[1:] - systoles[:-1])*res_x)
    Heart_Rate_dia = 60000.0 / (np.median(diastoles[1:] - diastoles[:-1])*res_x)
    Heart_Rate = (Heart_Rate_sys + Heart_Rate_dia) / 2.0

    return LVAW_s, LVAW_d, LVID_s, LVID_d, LVPW_s, LVPW_d, FS, LV_Mass, LV_Mass_Cor, Heart_Rate


def getRes(data):
    """
    Read from data from DICOM exported with regions
    
    Output: res_x (unit: s)
            res_y (unit: cm)
    """
    
    res_x = data.SequenceOfUltrasoundRegions[2].PhysicalDeltaX
    res_y = data.SequenceOfUltrasoundRegions[2].PhysicalDeltaY
    
    return res_x, res_y


def crop_frame(img):
    ''' Function to locate bounding box for actual image part'''
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sort = sorted(contours, key=cv2.contourArea, reverse=True)   
    x, y, w, h = cv2.boundingRect(sort[0])
    img_out = rgb2gray(img[y:y+h,x:x+w])
    return img_out