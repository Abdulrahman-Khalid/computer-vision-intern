import os
import cv2
import torch
import numpy as np

def get_cropped_frames(image_folder, ends_with=".jpg", start_point = (100, 50), end_point = (200, 150), padding = 25):
    images = [img for img in os.listdir(image_folder) if img.endswith(ends_with)]
    images = sorted(images)
    start_point_pad = (start_point[0]-padding, start_point[1]-padding)
    end_point_pad = (end_point[0]+padding, end_point[1]+padding)
    frames_cropped = []
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        frame = frame[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
        frames_cropped.append(frame)
    height_cropped, width_cropped, layers = frames_cropped[0].shape
    return frames_cropped, height_cropped, width_cropped

def load_image_rect_flatten_labels(img_path, start_point, end_point, padding,
                                    color = (0, 255, 0), color_pad = (255, 0, 0), thickness = 2):
    #Import image
    orig_image = cv2.imread(img_path)
    #Show the image with matplotlib
    start_point_pad = (start_point[0]-padding, start_point[1]-padding)
    end_point_pad = (end_point[0]+padding, end_point[1]+padding)    
    image = orig_image.copy()
    image = cv2.rectangle(image, start_point, end_point, color, thickness) 
    image = cv2.rectangle(image, start_point_pad, end_point_pad, color_pad, thickness) 
    # label image
    image_labels = np.zeros((orig_image.shape[0], orig_image.shape[1]))
    image_labels[start_point[1]:end_point[1], start_point[0]:end_point[0]] = 1.0
    # crop
    img = orig_image[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    labels = image_labels[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor(img)
    img = img.float()
    img = img.unsqueeze(0)
    labels = np.tensor(labels)
    flatten_labels = labels.view(-1)
    foreground = torch.where(flatten_labels == 1)[0]
    foreground = foreground.tolist()
    background = torch.where(flatten_labels == 0)[0]
    background = background.tolist()
    # Add to return image to see image with rectangles
    return img, flatten_labels, foreground, background, image 

def load_image_circle_flatten_labels(img_path, start_point, end_point, padding, 
                                      center_coordinates, radius, color = (0, 255, 0),
                                      color_pad = (255, 0, 0), thickness = 2):
    # Import image
    orig_image = cv2.imread(img_path)
    start_point_pad = (start_point[0]-padding, start_point[1]-padding)
    end_point_pad = (end_point[0]+padding, end_point[1]+padding)
    # Line thickness of 2 px 
    image = orig_image.copy()
    image = cv2.circle(image, center_coordinates, radius, color, thickness) 
    image = cv2.rectangle(image, start_point_pad, end_point_pad, color_pad, thickness) 
    # label image
    image_labels = np.zeros((orig_image.shape[0], orig_image.shape[1]))
    color = (255, 255, 255) 
    thickness = -1
    image_labels = cv2.circle(image_labels, center_coordinates, radius, color, thickness) 
    # crop
    img = orig_image[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    labels = image_labels[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor(img)
    img = img.float()
    img = img.unsqueeze(0)
    labels = torch.tensor(labels)
    flatten_labels = labels.view(-1)
    flatten_labels[flatten_labels==255] = 1.
    foreground = torch.where(flatten_labels == 1)[0]
    foreground = foreground.tolist()
    background = torch.where(flatten_labels == 0)[0]
    background = background.tolist()
    # Add to return image to see image with rectangles
    return img, flatten_labels, foreground, background, image

def load_image_cup_flatten_labels(img_path, start_point, end_point, start_point2,
                                   end_point2,  padding, cup_remove_label_rect=None, color = (0, 255, 0),
                                    color_pad = (255, 0, 0), thickness = 2):
    # Import image
    orig_image = cv2.imread(img_path)
    # Show the image with matplotlib
    start_point_pad = (start_point[0]-padding, start_point[1]-padding)
    end_point_pad = (end_point2[0]+padding, end_point[1]+padding)
    # Line thickness of 2 px 
    image = orig_image.copy()
    image = cv2.rectangle(image, start_point, end_point, color, thickness) 
    image = cv2.rectangle(image, start_point2, end_point2, color, thickness)
    if cup_remove_label_rect != None:
        unlabel_start_point, unlabel_end_point = cup_remove_label_rect
        image = cv2.rectangle(image, unlabel_start_point, unlabel_end_point, color_pad, thickness) 
    image = cv2.rectangle(image, start_point_pad, end_point_pad, color_pad, thickness) 
    # label image
    image_labels = np.zeros((orig_image.shape[0], orig_image.shape[1]))
    image_labels[start_point[1]:end_point[1], start_point[0]:end_point[0]] = 1.0
    image_labels[start_point2[1]:end_point2[1], start_point2[0]:end_point2[0]] = 1.0
    # unlabel image rect (make it 0)
    if cup_remove_label_rect != None:
        unlabel_start_point, unlabel_end_point = cup_remove_label_rect
        image_labels[unlabel_start_point[1]:unlabel_end_point[1], unlabel_start_point[0]:unlabel_end_point[0]] = 0.0
    # crop
    img = orig_image[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    labels = image_labels[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor(img)
    img = img.float()
    img = img.unsqueeze(0)
    labels = torch.tensor(labels)
    flatten_labels = labels.view(-1)
    foreground = torch.where(flatten_labels == 1)[0]
    foreground = foreground.tolist()
    background = torch.where(flatten_labels == 0)[0]
    background = background.tolist()
    # Add to return image to see image with rectangles
    return img, flatten_labels, foreground, background, image


def load_image_person_flatten_labels(img_path, rect_points_list, face_center, face_radius, padding_rect,
                                    color = (0, 255, 0), color_pad = (255, 0, 0), thickness = 2):
    #Import image
    orig_image = cv2.imread(img_path)
    #Show the image with matplotlib
    start_point_pad, end_point_pad = padding_rect
    image = orig_image.copy()
    image = cv2.circle(image, face_center, face_radius, color, thickness) 
    for rect_point in rect_points_list:
        start_point, end_point = rect_point
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    image = cv2.rectangle(image, start_point_pad, end_point_pad, color_pad, thickness) 
    # label image
    image_labels = np.zeros((orig_image.shape[0], orig_image.shape[1]))
    image_labels = cv2.circle(image_labels, face_center, face_radius, (255, 255, 255), -1) 
    for rect_point in rect_points_list:
        start_point, end_point = rect_point
        image_labels = cv2.rectangle(image_labels, start_point, end_point, (255, 255, 255), -1) 
    # crop
    img = orig_image[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    labels = image_labels[start_point_pad[1]:end_point_pad[1], start_point_pad[0]:end_point_pad[0]]
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor(img)
    img = img.float()
    img = img.unsqueeze(0)
    labels = torch.tensor(labels)
    flatten_labels = labels.view(-1)    
    flatten_labels[flatten_labels==255] = 1.
    foreground = torch.where(flatten_labels == 1)[0]
    foreground = foreground.tolist()
    background = torch.where(flatten_labels == 0)[0]
    background = background.tolist()
    # Add to return image to see image with rectangles
    return img, flatten_labels, foreground, background, image