import configparser
import os, errno
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import argparse
import PIL
from zipfile import ZipFile
from glob import glob
from tqdm import tqdm
import time
from tensorflow import keras
import json
import cutils
from bounding_box import bounding_box as bb
from config import *

from trusty.utils.smato_detector import *
from trusty.utils.network import *
from trusty.utils.utils_predict import *

from PIL import ImageFile


DOWNLOAD = None
INPUT_SIZE = 51
FONT = cv2.FONT_HERSHEY_SIMPLEX

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Predictor():
    """
        Class definition for the predictor.
    """
    def __init__(self, args, pifpaf_ver='tshufflenetv2k30'):
        device = args.device
        args.checkpoint = pifpaf_ver
        args.force_complete_pose = True
        if device != 'cpu':
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(device) if use_cuda else "cpu")
        else:
            self.device = torch.device('cpu')
        args.device = self.device
        # print('device : {}'.format(self.device))
        self.predictor_ = load_pifpaf(args)
        self.path_model = 'trusty/models/predictor'
        try:
            os.makedirs(self.path_model)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.get_model()
        if args.image_output is None:
            self.path_out = './output'
            self.path_out = filecreation(self.path_out)
        else:
            self.path_out = args.image_output
        self.args = args
        
        # persistent variables        
        self.vars = cutils.CUtils()
        self.vars.create_var('poses', {})
        self.vars.create_var('trust_fluc', {})
        self.vars.create_var('trust_smato', {})
        self.vars.create_var('trust_eye', {})
        self.vars.create_var('prev_trust', {})

    @staticmethod
    def scale_pose(keypoints, bbox):
        '''Returns a numpy array of keypoints scaled w.r.t the bounding box
        '''
        keypoints = np.array(keypoints).reshape((17, 3))[:, : -1]
        keypoints[:, 0] = (keypoints[:, 0] - bbox[0])/(bbox[2] - bbox[0])
        keypoints[:, 1] = (keypoints[:, 1] - bbox[1])/(bbox[3] - bbox[1])
        return keypoints

    def get_model(self):
        model = LookingModel(INPUT_SIZE)
        # print(self.device)
        model.load_state_dict(torch.load(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p'), map_location=self.device))
        model.eval()
        self.eye_model = model.to(self.device)
        self.smato_model = keras.models.load_model(SMATO_MODEL, custom_objects = {"f1": f1})

    def predict_look(self, boxes, keypoints, im_size, batch_wise = True):
        final_keypoints = []
        if batch_wise:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    final_keypoints.append(kps_final_normalized)
                tensor_kps = torch.Tensor([final_keypoints]).to(self.device)
                out_labels = self.eye_model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        else:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    tensor_kps = torch.Tensor(kps_final_normalized).to(self.device)
                    out_labels = self.eye_model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        return out_labels
    
    
    def render_image(self, image, data, keypoints, pred_labels, image_name, transparency):
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)

        for i, label in enumerate(pred_labels):
            if label > 0.7:
                color = (0, 255, 0)
            elif label > 0.4:
                color = (80, 127, 255)
            else:
                color = (0, 0, 255)

            mask = draw_skeleton(mask, keypoints[i], color)

        mask = cv2.erode(mask, (7, 7), iterations = 1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)
        
        # draw bounding boxes
        for dat in data:
            label = dat["trust"]
            if label > 0.7:
                color = "green"
            elif label > 0.4:
                color = "orange"
            else:
                color = "red"
            
            label = "Trust: {:.4f}".format(label)

            # label for smato
            if dat["trust_smato"] > 0.8:
                label += ", N"
            elif dat["trust_smato"] > 0.3:
                label += ", U"
            else:
                label += ", Y"

            bb.add(open_cv_image, *dat["bbox"][: -1], label, color = color)

        out_path = os.path.join(self.path_out, image_name[:-4] + '_predictions.png')
        cv2.imwrite(out_path, open_cv_image)

    def predict(self, array_im):
        args = self.args
        transparency = args.transparency

        loader = self.predictor_.images(array_im)

        for i, (pred_batch, _, meta_batch) in enumerate(loader):
            cpu_image = PIL.Image.open(array_im[i]).convert('RGB')
            
            pifpaf_outs = {
            'json_data': [ann.json_data() for ann in pred_batch],
            'image': cpu_image
            }

            im_name = os.path.basename(meta_batch['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])
            
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
            pred_labels = self.predict_look(boxes, keypoints, im_size)
            data = pifpaf_outs['json_data']

            # smartphone detection
            bboxes = [dat["bbox"] for dat in data]
            smato_detections = predict_smato(get_cropped_images(bboxes, cpu_image), self.smato_model)

            for k, label in enumerate(pred_labels):
                data[k]["trust_eye"] = float(label)
                data[k]["trust_smato"] = 1.0 - float(smato_detections[k])

                id_ = data[k]["id_"]

                # momentum for eye contact trust
                if id_ in self.vars.trust_eye:
                    self.vars.trust_eye[id_] = self.vars.trust_eye[id_] + data[k]["trust_eye"] * eye_rate
                    if self.vars.trust_eye[id_] > 1.00: self.vars.trust_eye[id_] = 1.00
                else:
                    self.vars.trust_eye[id_] = data[k]["trust_eye"]

                # momentum for smato trust
                if id_ in self.vars.trust_smato:
                    self.vars.trust_smato[id_] = self.vars.trust_smato[id_] * beta_smato + data[k]["trust_smato"] * (1 - beta_smato)
                else:
                    self.vars.trust_smato[id_] = data[k]["trust_smato"]

            # pose fluctuations
            for dat in data:
                id_ = dat["id_"]
                if id_ in self.vars.poses:
                    prev_pose = self.vars.poses[id_]
                    current_pose = self.scale_pose(dat["keypoints"], dat["bbox"])

                    # calculate trust based on pose fluctuations
                    mean_diff = np.mean(np.sqrt(np.sum(np.square(current_pose - prev_pose))))
                    trust_fluc = fluctuation_sensitvity/(mean_diff + 1e-4)
                    if trust_fluc > 1: trust_fluc = 1
                    dat["trust_fluc"] = trust_fluc
                    self.vars.trust_fluc[id_] = self.vars.trust_fluc[id_] * beta_fluc + trust_fluc * (1 - beta_fluc)
                else: 
                    dat["trust_fluc"] = 0.5
                    self.vars.trust_fluc[id_] = dat["trust_fluc"]

                # set previous pose point
                self.vars.poses[dat['id_']] = self.scale_pose(dat['keypoints'], dat["bbox"])

            # calculate final trust as a linear combination
            for dat in data:
                id_ = dat["id_"]
                dat["moving_trusts"] = {
                    "smato": self.vars.trust_smato[id_],
                    "eye": self.vars.trust_eye[id_],
                    "fluc": self.vars.trust_fluc[id_]
                }
                trust = alpha_trust["smato"] * self.vars.trust_smato[id_] + \
                        alpha_trust["eye"] * self.vars.trust_eye[id_] + \
                        alpha_trust["fluc"] * self.vars.trust_fluc[id_]
                if f_trust_inc == 0:
                    dat["trust"] = trust
                else:
                    if id_ in self.vars.prev_trust:
                        dat["trust"] = self.vars.prev_trust[id_] + f_trust_inc * trust
                    else:
                        dat["trust"] = initial_trust_ratio * trust
                    if dat["trust"] > 1.0: dat["trust"] = 1.0
                    
                    self.vars.prev_trust[id_] = dat["trust"]
                # print("Trust for id {}: {}".format(id_, dat["trust"]))

            # render image with labels
            self.render_image(pifpaf_outs['image'], data, keypoints, pred_labels, im_name, transparency)

            # save json output
            os.makedirs("jons/", exist_ok = True)
            numero = im_name[: -4]
            with open(f"jons/{numero}.json", "w") as outfile:
                json.dump(data, outfile, indent = 4)
        
        return data
