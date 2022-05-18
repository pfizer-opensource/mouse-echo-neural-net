# -*- coding: utf-8 -*-
"""
Main function for b-mode echo segmenter
Created on Fri Jun 26 10:49:25 2020

@author: DUANC01
"""

import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from skimage import io
import scipy
import argparse
import pydicom
import os
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import cv2

from util_bmode import dicom_preprocess, computeMetrics, findcardiacpeaks, getRes_rawDICOM, getRes, postprocess_masks
from util_nn_bmode import get_unet

if __name__ == '__main__':
    # Inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help="path to the input directory", metavar='input_dir', default='')
    parser.add_argument('--verbose', help="Save intermediate results for QC", action="store_true")
    args = parser.parse_args()

    if not args.input_dir:
        parser.error('No input directory provided, add --input_dir')

    input_dir = args.input_dir
    files = [file for file in os.listdir(input_dir) if file.endswith('.dcm')]

    # load trained model
    model = get_unet()
    model.load_weights('./model_weights/weights_ECHO_clean.h5')

    # Create output directory
    output_dir = os.path.join(input_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sys_areas = []
    dia_areas = []
    sys_volumes = []
    dia_volumes = []
    ejection_fractions = []
    # for saving images
    sys_segs = []
    dia_segs = []
    for file in files:
        filepath = os.path.join(input_dir, file)
        data = pydicom.dcmread(filepath)
        img_raw = data.pixel_array

        # read res (different method needed depending on how DICOM was saved)
        try:
            # unit is mm by default
            res_x, res_y = getRes_rawDICOM(data)
        except:
            # read res and convert cm to mm
            res_x, res_y = getRes(data)
            res_x, res_y = res_x * 10, res_y * 10

        images = []
        for i in range(img_raw.shape[0]):
            image = dicom_preprocess(img_raw[i])

            # Track original matrix size for tracking resolution
            org_y, org_x = image.shape

            # Resize to 256x256
            image = np.float32(image)
            image = np.uint8(np.round(resize(image, (256, 256))))

            images.append(image)

        imgs = np.stack(images)

        # Add new dimension
        imgs = imgs[..., np.newaxis]

        # Res changes after resize
        res_x, res_y = res_x * org_x / 256, res_y * org_y / 256

        # Standardize data using metrics from training set
        mean = 92
        std = 57
        imgs = imgs.astype(np.float32)
        imgs -= mean
        imgs /= std

        # predict
        # modified batch_size
        imgs_mask_test = model.predict(imgs, verbose=1, batch_size=1)

        # Convert data type
        imgs_mask_test = imgs_mask_test.astype(np.uint8)

        # Post-process masks - Chong Duan; Aug 11, 2021
        imgs_mask_test = postprocess_masks(imgs_mask_test, contappr=False)

        # cardiac phase
        systoles, diastoles = findcardiacpeaks(imgs_mask_test)

        # Compute metrics
        sys_area, dia_area, sys_volume, dia_volume, ef = computeMetrics(imgs_mask_test, systoles, diastoles, res_x,
                                                                        res_y)

        sys_areas.append(sys_area)
        dia_areas.append(dia_area)
        sys_volumes.append(sys_volume)
        dia_volumes.append(dia_volume)
        ejection_fractions.append(ef)

        # print('sys_area: {:.2f}'.format(sys_area))
        # print('dia_area: {:.2f}'.format(dia_area))
        # print('sys_volume: {:.2f}'.format(sys_volume))
        # print('dia_volume: {:.2f}'.format(dia_volume))
        # print('ef: {:.2f}'.format(ef))

        # For QC purpose
        if args.verbose:
            # print("Verbose QC is turned on")
            # Segmentation results
            sys_idx = systoles[0]
            dia_idx = diastoles[0]
            # systole phase
            a = rescale_intensity(imgs[sys_idx, :, :, 0], out_range=(-1, 1))
            b = (imgs_mask_test[sys_idx][:, :, 0]).astype('uint8')
            ab = mark_boundaries(a, b)
            ab = rescale_intensity(ab, out_range=(0, 255))
            ab = ab.astype('uint8')
            # io.imsave(os.path.join(output_dir, f'sys_{file}.png'), ab)
            sys_segs.append(ab)

            # diastole phase
            a = rescale_intensity(imgs[dia_idx, :, :, 0], out_range=(-1, 1))
            b = (imgs_mask_test[dia_idx][:, :, 0]).astype('uint8')
            ab = mark_boundaries(a, b)
            ab = rescale_intensity(ab, out_range=(0, 255))
            ab = ab.astype('uint8')
            # io.imsave(os.path.join(output_dir, f'dia_{file}.png'), ab)
            dia_segs.append(ab)

            # Save areas vs. time curves
            # areas = np.sum(np.reshape(imgs_mask_test, (imgs_mask_test.shape[0], -1)), axis=1)
            # areas_smoothed = scipy.ndimage.gaussian_filter(areas, 3)
            # plt.figure()
            # plt.plot(areas, 'b-')
            # plt.plot(areas_smoothed, 'r-')
            # plt.plot(systoles, areas_smoothed[systoles], 'g*')
            # plt.plot(diastoles, areas_smoothed[diastoles], 'gs')
            # plt.xlabel("Frame Number")
            # plt.ylabel("Pixel Counts")
            # plt.savefig(os.path.join(output_dir, f'CardiacPhases_{file}.png'))

    # Save output to spreadsheet
    files = [file[:-4] for file in files]
    df = pd.DataFrame(data={"animal_id": files, "sys_area (mm2)": sys_areas, "dia_area (mm2)": dia_areas,
                            "sys_volume (mm3)": sys_volumes, "dia_volume (mm3)": dia_volumes,
                            "ejection_fraction": ejection_fractions})
    df.to_csv(os.path.join(output_dir, 'output.csv'), index=False)

    # Save images to pdf if verbose is on
    if args.verbose:
        with PdfPages(os.path.join(output_dir, 'output_images.pdf')) as pdf:
            for i in range(len(files)):
                plt.figure(figsize=(8, 6))
                plt.suptitle(f'{files[i]}', fontsize=14)
                plt.subplot(121)
                plt.imshow(sys_segs[i])
                plt.title('systole')
                plt.subplot(122)
                plt.imshow(dia_segs[i])
                plt.title('diastole')
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

            # set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Segmentation Results'
            d['Author'] = 'Chong Duan'
            d['Subject'] = 'Systolic and diastolic segmentations'
            d['Keywords'] = 'Automated Echocardiography Analysis'
            d['CreationDate'] = datetime.datetime(2020, 9, 3)
            d['ModDate'] = datetime.datetime.today()
