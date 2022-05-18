# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:21:00 2021

@author: MONTGM11

Main function for m-mode analysis adapted from DUANC01's echoanalysis_main for
b-mode analysis
"""

import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
import argparse
import pydicom
import os
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from util_mmode import getRes_rawDICOM, compute_MMode_metrics, postprocess, getRes, crop_frame
from util_nn_mmode import get_unet

if __name__ == '__main__':
    # Cudnn environment
    #config = ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)

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
    model.load_weights('./model_weights/weights_MMode_clean_v4.h5')

    # Create output directory
    output_dir = os.path.join(input_dir, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Preallocate arrays to hold measurement results
    LVAWs_s = []
    LVAWs_d = []
    LVIDs_s = []
    LVIDs_d = []
    LVPWs_s = []
    LVPWs_d = []
    FSs = []
    LV_Masses = []
    LV_Mass_Cors = []
    Heart_Rates = []
    frames_used = []
    # For saving figures
    segs = []

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
            res_x, res_y = res_x * 1000, res_y * 10

        # Select frame to use
        if len(np.shape(img_raw)) == 4:
            ### ONE TO ONE TEST - FRAME SELECTION
            frame_select = range(np.shape(img_raw)[0])
            imslice = img_raw[frame_select]
        elif len(np.shape(img_raw)) == 3:
            imslice = img_raw
        else:
            continue
        images = []
        for z in range(np.shape(img_raw)[0]):
            # Select frame
            image = crop_frame(np.squeeze(imslice[z,:,:]))

            # Track original matrix size for tracking resolution
            org_y, org_x = image.shape

            # Resize to 256x256
            image = np.float32(image)
            image = rescale_intensity(image, out_range=(0, 255))
            image = np.uint8(np.round(resize(image, (256, 256))))
            images.append(image)

        imgs = np.stack(images)

        # Add new dimension
        imgs = imgs[..., np.newaxis]

        # Res changes after resize
        res_x, res_y = res_x * org_x / 256, res_y * org_y / 256

        # Standardize data using metrics from training set
        mean = 127
        std = 51
        imgs = imgs.astype(np.float32)
        imgs -= mean
        imgs /= std

        # predict
        # modified batch_size
        imgs_mask_test = model.predict(imgs, verbose=1, batch_size=1)

        # Calculate average "confidence" value for each frame
        conf = list(np.squeeze(np.mean(np.mean(np.mean(np.abs(imgs_mask_test-0.5),3),2),1)))

        # Round to nearest integer
        imgs_mask_test = np.round(imgs_mask_test)

        # Convert data type
        imgs_mask_test = imgs_mask_test.astype(np.uint8)

        # Apply post-processing algorithm to each label image
        label = np.squeeze(postprocess(imgs_mask_test))

        # Crop 10% from each side
        cutoff = int(np.floor(256*.1))
        label = label[:,:,cutoff-1:-cutoff]
        imgs = imgs[:,:,cutoff-1:-cutoff,:]


        # Compute metrics
        LVAW_s_tmp = []
        LVAW_d_tmp = []
        LVID_s_tmp = []
        LVID_d_tmp = []
        LVPW_s_tmp = []
        LVPW_d_tmp = []
        FS_tmp = []
        LV_Mass_tmp = []
        LV_Mass_Cor_tmp = []
        Heart_Rate_tmp = []
        for z in range(np.shape(label)[0]):
            LVAW_s, LVAW_d, LVID_s, LVID_d, LVPW_s, LVPW_d, FS, LV_Mass, LV_Mass_Cor, Heart_Rate = compute_MMode_metrics(np.squeeze(label[z,:,:]),res_y,res_x,agg_fn=np.median)
            LVAW_s_tmp.append(LVAW_s)
            LVAW_d_tmp.append(LVAW_d)
            LVID_s_tmp.append(LVID_s)
            LVID_d_tmp.append(LVID_d)
            LVPW_s_tmp.append(LVPW_s)
            LVPW_d_tmp.append(LVPW_d)
            FS_tmp.append(FS)
            LV_Mass_tmp.append(LV_Mass)
            LV_Mass_Cor_tmp.append(LV_Mass_Cor)
            Heart_Rate_tmp.append(Heart_Rate)

        # Select frame to use
        idx = conf.index(max(conf)) # Use frame with most confident predictions

        # Append measurements
        LVAWs_s.append(LVAW_s_tmp[idx])
        LVAWs_d.append(LVAW_d_tmp[idx])
        LVIDs_s.append(LVID_s_tmp[idx])
        LVIDs_d.append(LVID_d_tmp[idx])
        LVPWs_s.append(LVPW_s_tmp[idx])
        LVPWs_d.append(LVPW_d_tmp[idx])
        FSs.append(FS_tmp[idx])
        LV_Masses.append(LV_Mass_tmp[idx])
        LV_Mass_Cors.append(LV_Mass_Cor_tmp[idx])
        Heart_Rates.append(Heart_Rate_tmp[idx])
        frames_used.append(idx)

        # For QC purpose
        if args.verbose:
            # create image with segmentation results overlayed
            a = rescale_intensity(imgs[idx, :, :, 0], out_range=(-1, 1))
            b = label[idx,:,:]
            ab = mark_boundaries(a, b)
            ab = rescale_intensity(ab, out_range=(0, 255))
            ab = ab.astype('uint8')
            segs.append(ab)


    # Save output to spreadsheet
    files = [file[:-4] for file in files]
    #files = range(len(LVAWs_s))
    df = pd.DataFrame(data={"animal_id":files, "LVAW_sys (mm)": LVAWs_s, "LVAW_dia (mm)": LVAWs_d,
                            "LVPW_sys (mm)": LVPWs_s, "LVPW_dia (mm)": LVPWs_d, "LVID_sys (mm)": LVIDs_s,
                            "LVID_dia (mm)": LVIDs_d, "FS (%)": FSs, "LV_Mass (mg)": LV_Masses,
                            "LV_Mass_Cor (mg)": LV_Mass_Cors,
                            "Heart_Rate": Heart_Rates})
    df.to_csv(os.path.join(output_dir, 'output.csv'), index=False)

    # Save images to pdf if verbose is on
    if args.verbose:
        with PdfPages(os.path.join(output_dir, 'output_images.pdf')) as pdf:
            for i in range(len(segs)):
                plt.figure(figsize=(8, 6))
                #plt.title(f'{files[i]}', fontsize=14)
                plt.title(files[i], fontsize=14)
                plt.imshow(segs[i])
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

            # set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Segmentation Results'
            d['Author'] = 'Chong Duan'
            d['Subject'] = 'M-mode segmentations'
            d['Keywords'] = 'Automated Echocardiography Analysis'
            d['CreationDate'] = datetime.datetime(2021, 1, 21)
            d['ModDate'] = datetime.datetime.today()
