import colorsys
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import *

import shutil
import json
import time




class YOLO(object):
    _defaults = {
        "model_path": 'model_data/ep079-loss10.104-val_loss14.523.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        #self.sess = K.get_session()
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def box_prediction(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        return out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()

        

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:  for i=numel(mpre)-1:-1:1
                                mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


    


def get_mAP(annotation_lines, class_names, score_threshold=0.25, iou_threshold=0.50):
    MINOVERLAP = 0.3 # default value (defined in the PASCAL VOC2012 challenge)
    
    num_classes = len(class_names)
    num_samples = len(annotation_lines)
    
    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists('mAP'): os.mkdir('mAP')
    os.mkdir(ground_truth_dir_path)
    
    print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')
    
    gt_counter_per_class = {}
    for index in range(num_samples):
        annotation_line = annotation_lines[index]
        #original_image, bbox_data_gt = get_random_data(annotation_line, input_shape, random=True)
        line = annotation_line.split()
        bbox_data_gt = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
 
        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            
        num_bbox_gt = len(bboxes_gt)

        bounding_boxes = []
        
        for i in range(num_bbox_gt):
            class_name = class_names[classes_gt[i]]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
                
            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)


    gt_classes = list(gt_counter_per_class.keys())
    # sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    
    
    times = []
    json_pred = [[] for i in range(n_classes)]
    
    yolo = YOLO()
    
    for index in range(num_samples):
        annotation_line = annotation_lines[index]
        line = annotation_line.split()
        original_image_path = line[0]
        original_image = Image.open(original_image_path)
        
        t1 = time.time()

        out_boxes, out_scores, out_classes = yolo.box_prediction(original_image)
        print(index)
        
        t2 = time.time()
        
        times.append(t2-t1)
        
        #for i, c in reversed(list(enumerate(out_classes))):
        for i, c in (enumerate(out_classes)):
            class_name = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            xmin, ymin, xmax, ymax = box
            #bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bbox = str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
            #json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})
            json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": bbox})
            
    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    
    for class_name in gt_classes:
        json_pred[gt_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_classes.index(class_name)], outfile)
            
    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    
    # open file to store the results
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
            predictions_data = json.load(open(predictions_file))

            # Assign predictions to ground truth objects
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            # Write to results.txt
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)
        
        return mAP*100
        
        

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
    
if __name__ == '__main__':
    annotation_path = 'test_set.txt'
    classes_path = 'model_data/voc_classes.txt'
    class_names = get_classes(classes_path)
    input_shape = (416,416)
    
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    
    
    mAP = get_mAP(annotation_lines, class_names, score_threshold=0.3, iou_threshold=0.45)
    print('mean average precision: {mAP:0.3f}'.format(mAP=mAP))
    
    #with open("mAP/results.txt") as results_file
    #    result_lines = results_file.readlines()
        
    #print(result_lines)