import os
from os import getcwd
import xml.etree.ElementTree as ET


train_image_path = './trainval/VOCdevkit/VOC2012/JPEGImages/'
train_xml_path = './trainval/VOCdevkit/VOC2012/Annotations/'

#test_image_path = './test/VOCdevkit/VOC2007/JPEGImages/'
#test_xml_path = './test/VOCdevkit/VOC2007/Annotations/'

trainval_file = 'trainval_set_2012.txt'
#test_file = 'test_set.txt'


train_image_directory = os.listdir(train_image_path)
train_images = [train_image_path + image for image in train_image_directory]

train_xml_directory = os.listdir(train_xml_path)
train_xmls = [train_xml_path + xml for xml in train_xml_directory]

#test_image_directory = os.listdir(test_image_path)
#test_images = [test_image_path + image for image in test_image_directory]

#test_xml_directory = os.listdir(test_xml_path)
#test_xmls = [test_xml_path + xml for xml in test_xml_directory]

working_directory = getcwd()


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def processing_dataset(image_fullname, xml_fullname, file_name):
    xml_file = open(xml_fullname)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    row = ""
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in classes:
            continue
            
        class_id = classes.index(name)
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        
        row = row + " " + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + str(class_id)
        
    if row != "":
        list_file = open(file_name, 'a')
        file_string = working_directory + str(image_fullname)[1:] + row + '\n'
        list_file.write(file_string)
        list_file.close()


for i in range(len(train_xmls)):
    train_image_fullname = train_images[i]
    train_xml_fullname = train_xmls[i]
    processing_dataset(train_image_fullname, train_xml_fullname, trainval_file)
    
    

#for j in range(len(test_xmls)):
#    test_image_fullname = test_images[j]
#    test_xml_fullname = test_xmls[j]
#    processing_dataset(test_image_fullname, test_xml_fullname, test_file)