import os
import json
import random
from random import randint
import numpy as np

# APC Dataset Creation

def create_pairs(x, digit_indices , num_classes):
	'''
	Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''
	pairs = []
	labels = []
	for d in range(num_classes):
	    max_num_d = len(digit_indices[d]) - 1
	    for i in range(max_num_d):
	        z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
	        pairs += [[x[z1], x[z2]]]
	        inc = random.randrange(1, num_classes)
	        dn = (d + inc) % num_classes
	        max_num_dn = len(digit_indices[dn]) - 1
	        dn_index = i
	        if(dn_index > max_num_dn):
	            if(max_num_dn == 0):
	                dn_index = max_num_dn
	            else:
	                dn_index = random.randrange(0, max_num_dn)
	        z1, z2 = digit_indices[d][i], digit_indices[dn][dn_index]
	        pairs += [[x[z1], x[z2]]]
	        labels += [0, 1.]
	return np.array(pairs), np.array(labels).astype(np.float32)


def generate_apc_datapairs(src_path , img_file, label_file , num_classes):
	with open(os.path.join(src_path , img_file)) as f:
		X = f.read().splitlines()
	with open(os.path.join(src_path , label_file)) as f:
		y = f.read().splitlines()
	X = np.array(X)
	y = np.array(y)
	y = y.astype(np.uint8)
	y = y - 1

	digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
	pairs, labels = create_pairs(X, digit_indices, num_classes)
	return pairs, labels

# Inhouse Dataset Creation

def write_data_struct(src_path, dirs, output_filename):
	valid_images = [".jpg",".png"]
	data = []
	for directory in dirs:
		img_path = os.path.join(src_path, directory)
		dir_data = {}
		dir_data["class"] = directory
		dir_images = []
		for f in sorted(os.listdir(img_path)):
			file_ext = os.path.splitext(f)[1]
			file_name = os.path.splitext(f)[0]
			if file_ext.lower() not in valid_images:
				continue
			dir_images.append(str(os.path.join(src_path, directory , f)))
		dir_data["images"] = dir_images
		data.append(dir_data)
	with open(os.path.join(src_path,  output_filename), 'w') as outfile:
            json.dump(data, outfile)


def generate_inhouse_datapairs(src_path , dataset="train"):
	if dataset == "test":
		dataset_file_name = "test.json"
		dirs = ["27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44"]
		write_data_struct(src_path, dirs, dataset_file_name)
	else:
		dataset_file_name = "train.json"
		dirs = ["3","4","5","6","7","9","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26"]
		write_data_struct(src_path, dirs, dataset_file_name)
	pairs = []
	labels = []
	with open(os.path.join(src_path , dataset_file_name)) as data_file:    
			dataset = json.load(data_file)

	num_classes = len(dataset)
	for class_idx, obj_class_data in enumerate(dataset):
		class_name = obj_class_data["class"]
		class_imgs = obj_class_data["images"]
		num_images = len(class_imgs)
		group_size = 3
		
		for i in xrange(group_size-1, num_images, group_size):
			pairs += [[class_imgs[i], class_imgs[i-1]]]
			pairs += [[class_imgs[i-1], class_imgs[i-2]]]
			pairs += [[class_imgs[i], class_imgs[i-2]]]
			labels += [0., 0., 0.]

		for class_img in class_imgs:
			random_range = range(0,class_idx-1) + range(class_idx+1,num_classes-1)
			random_class_index = random.choice(random_range)
			random_img_index = randint(0, len(dataset[random_class_index]["images"])-1)
			pairs += [[class_img, dataset[random_class_index]["images"][random_img_index]]]
			labels += [1.]
	return np.array(pairs), np.array(labels)


if __name__ == "__main__":
	pairs, labels = generate_inhouse_datapairs("/home/dexf17/Work/HDD/Data/RCNN/30OBJ/Siamese" , "train")
	print pairs[100]
	pairs, labels = generate_apc_datapairs("data" , "train-product-imgs.txt", "train-product-labels.txt", 41)
	print pairs.shape
