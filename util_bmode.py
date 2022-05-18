#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:25:30 2020

@author: duanc01
"""

import numpy as np
import cv2
import scipy

import matplotlib.pyplot as plt

def echo_preprocess(img, img_annotated):
    """
    Input:
        img: [heightxwidthx3], VevoLAB image w/o annotations
        img_annotated: [heightxwidthx3], VevoLAB image w/ annotations
    
    Output:
        img1: echo image
        mask_img: the LV binary mask (1 indicates LV, 0 indicates other tissues), same size as img1
    """
    
    # Identify Echo Image Part (i.e., rectanglular bounding box)
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img_annotated, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sort = sorted(contours, key=cv2.contourArea, reverse=True)   
    x, y, w, h = cv2.boundingRect(sort[0])
    
    img1 = img[y:y+h,x:x+w,:]
    img2 = img_annotated[y:y+h,x:x+w,:]
    
    # Mask of all annotions
    mask = np.logical_not(np.logical_and(img2[:,:,0] == img2[:,:,1], img2[:,:,1] == img2[:,:,2]))
    
    # Chong Duan, March 30, 2020
    # Check if there is any label/annotations on img2. If not, do not proceed with the rest of processing
    if np.sum(mask) < 5:
        return img1, mask
    
    # Identify the LV contour
    contours,hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea, reverse=True)   
    
    # Center of Contour
    x_center, y_center = (np.mean(np.squeeze(sort[0]),axis=0)).astype(np.uint32)
    
    # Draw the LV contour (largest one), and flood fill the contour
    mask_img = np.zeros(img2.shape, dtype=np.uint8)
    cv2.drawContours(mask_img, sort[0], -1, (255, 255, 255), 3)
    h, w = mask_img.shape[:2]
    mask4fill = np.zeros((h+2, w+2), np.uint8) # this mask needs to be 2 pixel larger
    cv2.floodFill(mask_img, mask4fill, (x_center,y_center), (255,255,255))
    
    # Convert to Grayscale (all 3 channels are the same)
    img1 = img1[:,:,0]
    mask_img = mask_img[:,:,0]
    mask_img[np.where(mask_img==255)] = 1
    
    return img1, mask_img

def dicom_preprocess(img):
    """
    Extract the echo image part from a raw DICOM frame
    """
    # Identify Echo Image Part (i.e., rectanglular bounding box)
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sort = sorted(contours, key=cv2.contourArea, reverse=True)   
    x, y, w, h = cv2.boundingRect(sort[0])
    
    echo_img = img[y:y+h,x:x+w,:]
    return echo_img[:,:,0]

def findLongAxis(mask):
    """
    Find the long axis of the input: lv mask
    
    Return two points defining the long axis: pt1, pt2
    
    """
    # Identify the LV contour
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea, reverse=True)   
    
    # Take the largest contour
    cont = np.squeeze(sort[0])
    dists = np.max(np.sqrt(np.sum((cont - cont[:,np.newaxis])**2, axis=2)), axis=0)
    
    idx_max = np.argmax(dists)
    pt1 = cont[idx_max]
    
    # Search for pt2
    count_diff = []
    for pt in cont:
        ptA = pt1
        ptB = pt
        if ptA[0] == ptB[0] and ptA[1] == ptB[1]:
            count_diff.append(np.inf)
            continue
        count_above = 0
        count_below = 0
        for point in cont:
            val = (point - ptA)[0]*(point - ptB)[1] - (point - ptA)[1]*(point - ptB)[0]
            if val > 0:
                count_above += 1
            elif val < 0:
                count_below += 1
        
        count_diff.append(count_above - count_below)
    
    count_diff = np.abs(np.array(count_diff))
    
    idx2 = np.argmin(count_diff)
    pt2 = cont[idx2]
    
    return pt1, pt2


def computeDist(pt1, pt2):
    return np.sqrt(np.sum((pt1 - pt2)**2))


def findcardiacphase(labels):
    """
    To find all systole phase and diastole phase
    
    """
    areas = np.sum(np.reshape(labels, (labels.shape[0],-1)),axis=1)
    
    # smooth the areas curve; need to adjust the window size parameter
    areas = scipy.ndimage.gaussian_filter(areas, 5)
    
    # a phase that has area smaller than either side
    systoles = np.where(np.r_[True, areas[1:] <= areas[:-1]] & np.r_[areas[:-1] <= areas[1:], True])

    # a phase that has area larger than either side
    diastoles = np.where(np.r_[True, areas[1:] >= areas[:-1]] & np.r_[areas[:-1] >= areas[1:], True])

    return systoles[0], diastoles[0]


def findcardiacpeaks(labels):
    """
    To find all systole phase and diastole phase with scipy.signal_find_peaks func
    
    """
    areas = np.sum(np.reshape(labels, (labels.shape[0],-1)),axis=1)
    
    # smooth the areas curve; need to adjust the window size parameter
    areas = scipy.ndimage.gaussian_filter(areas, 5)
    
    diastoles = scipy.signal.find_peaks(areas, distance=10)
    diastoles = diastoles[0]
    systoles = scipy.signal.find_peaks(areas * -1, distance=10)
    systoles = systoles[0]
    
    # when only one cardiac cycle is labeled
    if len(diastoles) == 0:
        diastoles = [np.argmax(areas)]
        
    if len(systoles) == 0:
        systoles = [np.argmin(areas)]
    
    return systoles, diastoles


def computeEF(labels, systoles, diastoles):
    """
    """
    
    sys_labels = labels[systoles,:,:]
    dia_labels = labels[diastoles,:,:]      
    
    sys_volumes = []
    for sys_label in sys_labels:
        pt1, pt2 = findLongAxis(sys_label)
        L = computeDist(pt1, pt2)
        Area = np.sum(sys_label)
        Area = Area.astype(np.float64)
        Volume = Area * Area / L
        sys_volumes.append(Volume)
    
    sys_volume_mean = np.mean(np.array(sys_volumes))
    
    dia_volumes = []
    for dia_label in dia_labels:
        pt1, pt2 = findLongAxis(dia_label)
        L = computeDist(pt1, pt2)
        Area = np.sum(dia_label)
        Area = Area.astype(np.float64)
        Volume = Area * Area / L
        dia_volumes.append(Volume)
    
    dia_volume_mean = np.mean(np.array(dia_volumes))
    
    return (dia_volume_mean - sys_volume_mean)/dia_volume_mean


def computeMetrics(labels, systoles, diastoles, res_x, res_y):
    """
    Parameters
    ----------
    labels : TYPE
        DESCRIPTION.
    systoles : TYPE
        DESCRIPTION.
    diastoles : TYPE
        DESCRIPTION.
    res_x : TYPE
        DESCRIPTION.
    res_y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # remove the last dimension of labels
    labels = np.squeeze(labels)
    
    sys_labels = labels[systoles,:,:]
    dia_labels = labels[diastoles,:,:]
    
    sys_areas = []
    sys_volumes = []
    for sys_label in sys_labels:
        pt1, pt2 = findLongAxis(sys_label)
        
        # long-axis length
        pt1 = pt1 * np.array([res_x, res_y])
        pt2 = pt2 * np.array([res_x, res_y])
        L = computeDist(pt1, pt2)  # unit is mm
        
        # Area
        Area = np.sum(sys_label) * res_x * res_y    # unit is mm2
        Area = Area.astype(np.float64)
        sys_areas.append(Area)
        
        # Area-Length method
        Volume = 8.0/(3*np.pi) * Area * Area / L    # unit is mm3 or uL
        sys_volumes.append(Volume)
    
    sys_area_mean = np.mean(np.array(sys_areas))
    sys_volume_mean = np.mean(np.array(sys_volumes))
    
    dia_areas = []
    dia_volumes = []
    for dia_label in dia_labels:
        pt1, pt2 = findLongAxis(dia_label)
        
        # Long-Axis Length
        pt1 = pt1 * np.array([res_x, res_y])
        pt2 = pt2 * np.array([res_x, res_y])
        L = computeDist(pt1, pt2)  # unit is mm
        
        # Area
        Area = np.sum(dia_label) * res_x * res_y
        Area = Area.astype(np.float64)
        dia_areas.append(Area)
        
        # Area-Length method for volume
        Volume = 8.0/(3*np.pi) * Area * Area / L
        dia_volumes.append(Volume)
    
    dia_area_mean = np.mean(np.array(dia_areas))
    dia_volume_mean = np.mean(np.array(dia_volumes))
    
    # unit %
    ef = (dia_volume_mean - sys_volume_mean)/dia_volume_mean * 100
    
    return sys_area_mean, dia_area_mean, sys_volume_mean, dia_volume_mean, ef


def getRes(data):
    """
    Input: data (pydicom.dataset.FileDataset), read from analyzed DICOM
    
    Output: res_x (unit: cm?)
            res_y (unit: cm?)
    """
    
    res_x = data.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
    res_y = data.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
    
    return res_x, res_y


def getRes_rawDICOM(data):
    """
    Input: data (pydicom.dataset.FileDataset), read from raw DICOM
    
    Output: res_x (unit: mm)
            res_y (unit: mm)
    """
    
    res_y = float(data.PixelSpacing[0])
    res_x = float(data.PixelSpacing[1])
    
    return res_x, res_y


def postprocess_masks(masks, contappr=False):
    """
    To remove small regions in the network output masks

    masks: output of U-Net segmentation; NxHxWx1
    """
    out = masks.copy()
    for i in range(masks.shape[0]):
        mask = masks[i, :, :, 0]
        # Identify the LV contour
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        sort = sorted(contours, key=cv2.contourArea, reverse=True)

        # Some masks are empty, skip those masks - Chong Duan; Oct 21, 2020
        if len(sort) < 1:
            continue

        cont = sort[0]

        # Contour approximation
        if contappr:
            epsilon = 0.01 * cv2.arcLength(cont, True)
            cont = cv2.approxPolyDP(cont, epsilon, True)

        # Fill the contour to get a mask
        # Center of Contour
        x_center, y_center = (np.mean(np.squeeze(cont), axis=0)).astype(np.uint32)

        # Draw the LV contour (largest one), and flood fill the contour
        mask_img = np.zeros(mask.shape, dtype=np.uint8)
        # cv2.drawContours takes a list of contours, -1 flags draw all of them
        cv2.drawContours(mask_img, [cont], -1, (1, 1, 1), 1)

        h, w = mask_img.shape[:2]
        mask4fill = np.zeros((h + 2, w + 2), np.uint8)  # this mask needs to be 2 pixel larger
        cv2.floodFill(mask_img, mask4fill, (x_center, y_center), (1, 1, 1))
        out[i, :, :, 0] = mask_img
    return out