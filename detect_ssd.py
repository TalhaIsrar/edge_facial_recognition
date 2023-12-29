import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import tensorflow as tf
import numpy as np
from PIL import Image as conv_img
from tensorflow.keras.models import model_from_json

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def pil_img_to_encoding(in_img, model):
    im = conv_img.fromarray(in_img)
    newsize = (160, 160)
    img = im.resize(newsize) 
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


def who_is_it(image_path, database1, database2, database3, model):
    encoding = pil_img_to_encoding(image_path,model)
    min_dist = 100
    for (name, db_enc) in database1.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
            if min_dist < 0.55:
                return min_dist, identity, True
    
    if min_dist > 0.6:
        for (name, db_enc) in database2.items():
            dist = np.linalg.norm(encoding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
                if min_dist < 0.55:
                    return min_dist, identity, True

    if min_dist > 0.6:
        for (name, db_enc) in database3.items():
            dist = np.linalg.norm(encoding - db_enc)
            if dist < min_dist:
                min_dist = dist
                identity = name
                if min_dist < 0.55:
                    return min_dist, identity, True

    exists = False
        
    if min_dist > 0.80:
        print("Not in the database." + " Min dist is " + str(min_dist))
        exists = False
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        exists = True
        
    return min_dist, identity, exists

def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    model_json = "ssd/facenet/model.json"
    model_h5 = "ssd/facenet/model.h5"
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    my_model = model_from_json(loaded_model_json)
    my_model.load_weights(model_h5)
    FRmodel = my_model
    ssd_images = "ssd"
    database1 = {}
    for images1 in os.listdir(ssd_images+"/images/db1"):
    
        # check if the image ends with jpg
        if (images1.endswith(".jpg")):
            database1[images1.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

        if (images1.endswith(".png")):
            database1[images1.replace('.png','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

        if (images1.endswith(".JPG")):
            database1[images1.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

        if (images1.endswith(".jpeg")):
            database1[images1.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db1/" + images1, FRmodel)

    
    database2 = {}
    for images2 in os.listdir(ssd_images+"/images/db2"):
    
        # check if the image ends with jpg
        if (images2.endswith(".jpg")):
            database2[images2.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

        if (images2.endswith(".png")):
            database2[images2.replace('.png','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

        if (images2.endswith(".JPG")):
            database2[images2.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

        if (images2.endswith(".jpeg")):
            database2[images2.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db2/" + images2, FRmodel)

    
    database3 = {}
    for images3 in os.listdir(ssd_images+"/images/db3"):
    
        # check if the image ends with jpg
        if (images3.endswith(".jpg")):
            database3[images3.replace('.jpg','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

        if (images3.endswith(".png")):
            database3[images3.replace('.png','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

        if (images3.endswith(".JPG")):
            database3[images3.replace('.JPG','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

        if (images3.endswith(".jpeg")):
            database3[images3.replace('.jpeg','')] = img_to_encoding(ssd_images+"/images/db3/" + images3, FRmodel)

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            imc = im0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    crop = save_one_box(xyxy, imc, BGR=True)  
                    dist, name, exists = who_is_it(crop, database1, database2, database3, FRmodel)
                    c = int(cls)  # integer class
                    label = f'{name} {dist:.2f}' if exists else f'Unidentified {dist:.2f}'
                    kpts = det[det_index, 6:]
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

            t3 = time_synchronized()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t3 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=160, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', type=int, default=5, help='number of keypoints')
    opt = parser.parse_args()
    #check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
