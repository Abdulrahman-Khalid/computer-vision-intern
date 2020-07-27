from tqdm import tqdm
import torch
import cv2
import time
import seaborn as sns
import pickle
import numpy as np
import random as rd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
from skimage import io

def remove_folder_content(folder='./heatmaps/'):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def save_centroids(path, centroids):
    with open(path,'wb') as f:
        pickle.dump(centroids, f)
def load_centroids(path):
    centroids = None
    with open(path,'rb') as f:
        centroids = pickle.load(f)
    return centroids

def get_heat_map(embeddings, shape, centroids, num_classes=2):
    heat_map = np.zeros(len(embeddings))
    for emb_idx, embedding in enumerate(embeddings):
        min_dist = np.inf
        for class_idx, centriod in enumerate(centroids):
            dist = np.linalg.norm(centriod - embedding)
            if(dist < min_dist):
                min_dist = dist
                heat_map[emb_idx] = class_idx
    heat_map = heat_map.reshape(shape)
    return heat_map

def get_real_heatmap(embeddings, shape, centroids, axis = 1): # axis = 2 if 3D
    
    since = time.time_ns()
    
    dist_background = np.linalg.norm(embeddings - centroids[0], axis=axis)
    dist_foreground = np.linalg.norm(embeddings - centroids[1], axis=axis)
    heat_map = dist_background / dist_foreground
    heat_map = heat_map.reshape(shape)
    
    now = time.time_ns()
    heat_map_time = (now - since) * 10e-9
    
    return heat_map, heat_map_time

def get_heatmaps(model, centroids, frames, image_shape, frames_step=1, num_classes=2, use_gpu=True):
    since = time.time()
    model.eval()
    if use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    heat_maps = []
    frames_cpy = frames.copy()
    with torch.set_grad_enabled(False):
        for idx in tqdm(range(0, len(frames), frames_step), desc="Generate Heat Maps"):            
            frames_cpy[idx] = np.moveaxis(frames_cpy[idx], -1, 0)
            frames_cpy[idx] = torch.tensor(frames_cpy[idx])
            frames_cpy[idx] = frames_cpy[idx].float()
            frames_cpy[idx] = frames_cpy[idx].unsqueeze(0)
            if use_gpu:
                frames_cpy[idx] = frames_cpy[idx].to(device)
            embeddings = model(frames_cpy[idx])
            if use_gpu:
                embeddings = embeddings.data.cpu().numpy()
            else:
                embeddings = embeddings.data.numpy()
                            
            heat_map = get_heat_map(embeddings, image_shape, centroids, num_classes)
            heat_maps.append(heat_map)
    time_elapsed = time.time() - since
    print('Time Elapsed (get_heatmaps func) {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return heat_maps

def get_real_heatmaps(model, centroids, frames, image_shape, frames_step=1, num_classes=2, use_gpu=True, axis=1):
    since = time.time()
    model.eval()
    if use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    heat_maps = []
    frames_cpy = frames.copy()
    
    time_model_list = []
    time_heat_list = []
    time_total_list = []
    
    with torch.set_grad_enabled(False):
        for idx in tqdm(range(0, len(frames), frames_step), desc="Generate Heat Maps"):
            
            frame_start_time = time.time_ns()

            frames_cpy[idx] = np.moveaxis(frames_cpy[idx], -1, 0)
            frames_cpy[idx] = torch.tensor(frames_cpy[idx])
            frames_cpy[idx] = frames_cpy[idx].float()
            frames_cpy[idx] = frames_cpy[idx].unsqueeze(0)
            if use_gpu:
                frames_cpy[idx] = frames_cpy[idx].to(device)
            embeddings = model(frames_cpy[idx])
            if use_gpu:
                embeddings = embeddings.data.cpu().numpy()
            else:
                embeddings = embeddings.data.numpy()
            
            now = time.time_ns()
            model_time = (now - frame_start_time) * 10e-9
            
            time_model_list.append(model_time)

            heat_map, heat_map_time = get_real_heatmap(embeddings, image_shape, centroids, axis=axis)
            heat_maps.append(heat_map)
            
            time_heat_list.append(heat_map_time)
            
            total_frame_time = model_time + heat_map_time
            time_total_list.append(total_frame_time)
    time_elapsed = time.time() - since
    print('Time Elapsed (get_heatmaps func) {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    time_model_list = np.array(time_model_list)
    time_heat_list = np.array(time_heat_list)
    time_total_list = np.array(time_total_list)
    
    print('Model time mean =', np.mean(time_model_list, axis=0))
    print('Model time std =', np.std(time_model_list, axis=0))

    print('Heatmap time mean =', np.mean(time_heat_list, axis=0))
    print('Heatmap time std =', np.std(time_heat_list, axis=0))

    print('Total time mean =', np.mean(time_total_list, axis=0))
    print('Total time std =', np.std(time_total_list, axis=0))

    return heat_maps
    
def create_video(heat_maps, video_path, width, height, froucc=0, fps=1):
    since = time.time()
    video = cv2.VideoWriter(video_path, froucc, fps, (width, height), 0)
    for heat_map in heat_maps:
        video.write((heat_map*255).astype('uint8'))
    video.release()
    time_elapsed = time.time() - since
    print('Time Elapsed (create_video func) {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def easy_create_video(model, centroids, frames, video_path, width, height, froucc=0, use_gpu=True, fps=1, frames_step=1, num_classes=2, ret_heat_maps=False, real_heat_map=False):
    since = time.time()
    if real_heat_map:
        heat_maps = get_real_heatmaps(model, centroids, frames, (height, width), frames_step, num_classes, use_gpu)
    else:   
        heat_maps = get_heatmaps(model, centroids, frames, (height, width), frames_step, num_classes, use_gpu)
    create_video(heat_maps, video_path, width, height, froucc, fps)
    time_elapsed = time.time() - since
    print('Time Elapsed (easy_create_video func) {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if ret_heat_maps: return heat_maps

def easy_create_real_video(model, centroids, frames, video_path, width, height,
                            froucc=cv2.VideoWriter_fourcc('M','J','P','G'), use_gpu=True,
                            fps=1, frame_range=(0, 300), frames_step=1, num_classes=2, ret_heat_maps=False,
                            heatmap_path = 'heatmaps/heatmap', img_ext = '.jpg', vmin=0, vmax=2, axis=1):
    real_heat_maps = get_real_heatmaps(model, centroids, frames[frame_range[0]:frame_range[1]], (height, width), frames_step=frames_step, num_classes=num_classes, use_gpu=use_gpu, axis=axis)
    for idx, heat_map in enumerate(real_heat_maps):
        sns.heatmap(heat_map, cmap = 'jet', center=(vmin + vmax) / 2., vmax=vmax)
        plt.savefig(heatmap_path + str(idx) + img_ext, dpi=200)
        plt.clf() # clear figure (v.imp)    
        
    f_idx = 0
    for idx in range(len(real_heat_maps)):
        saved = io.imread(heatmap_path + str(idx) + img_ext)
        grouped_image=np.zeros([saved.shape[0], frames[f_idx].shape[1]+saved.shape[1],3])
        grouped_image[:frames[f_idx].shape[0],:frames[f_idx].shape[1]] = frames[f_idx]
        grouped_image[:,frames[f_idx].shape[1]:] = saved[:,:,0:3]
        io.imsave(heatmap_path + str(idx) + img_ext,grouped_image.astype('uint8'))
        f_idx += frames_step

    images = [heatmap_path + str(i) + img_ext for i in range(len(real_heat_maps))]
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    for image in images:
        img =  cv2.imread(image)
        img = cv2.resize(img, (width,height), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(image, img)

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    video = cv2.VideoWriter(video_path, froucc, fps, (width,height))
    for image in images:
        video.write(cv2.imread(image))
    video.release()

    if ret_heat_maps: return real_heat_maps

def plot_PCA(embeddings, targets, out_features_size, classes, flat_first=False, is_3D=False,colors=None, pca_title='My PCA Graph', xlim=None, ylim=None, draw_PCA_per_varc=False, figure_size=(10,10), top_range_loadingfactors=None):
    # Generate random colors for each class
    if colors == None:
        colors = []
        r = lambda: rd.randint(0, 200) # (0, 255) not 255 to avoid colors diff than white
        for _ in range(len(classes)):
            colors.append('#%02X%02X%02X' % (r(),r(),r()))
    assert(len(colors) == len(classes))

    # First center and scale the data
    # After centering avg value for each gene will be 0
    # After scaling std for each gene will be 1
    embeddings_indecies = ['embedding' + str(i) for i in range(1, len(embeddings[0])+1)]
    df = pd.DataFrame(columns=embeddings_indecies, index=np.arange(1, len(embeddings)+1))
    for i in range(out_features_size):
        df.loc[:, embeddings_indecies[i]] = embeddings[:, i]
    # print(df.head())
    # print("Shape:", df.shape)
                          
    # scaled_data = preprocessing.scale(df.T) if(samples_are_columns) else preprocessing.scale(df)
    scaled_data = preprocessing.scale(df)
    pca = PCA()
    pca.fit(scaled_data) # calc loading scores for every pca
    pca_data = pca.transform(scaled_data) # generate coordinates    
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1) # Percentage of Explained Variance in %
    # labels to PCA number corresponding to Percentage of Explained Variance
    labels = ['PC' + str(i) for i in range(1, len(per_var)+1)] 
    
    # plot scree plot of loading factors 
    if draw_PCA_per_varc:
        plt.figure(figsize=figure_size)
        plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.show()
    
    #the following code makes a fancy looking plot using PC1 and PC2
    pca_df = pd.DataFrame(pca_data, index=np.arange(1, len(embeddings)+1), columns=labels)
    # print(pca_df)
    if is_3D:
        ax = plt.figure(figsize=figure_size)
        ax = Axes3D(ax)
    else:
        plt.figure(figsize=figure_size)
    for i in range(len(classes)):
        inds = np.where(targets == i)[0]
        if is_3D:
            ax.scatter(pca_df.iloc[inds, 0], pca_df.iloc[inds, 1], pca_df.iloc[inds, 2], alpha=0.5, color=colors[i])
        else:
            plt.scatter(pca_df.iloc[inds, 0], pca_df.iloc[inds, 1], alpha=0.5, color=colors[i])
    if is_3D:
        if xlim:
            ax.xlim(xlim[0], xlim[1])
        if ylim:
            ax.ylim(ylim[0], ylim[1])

        ax.set_title(pca_title)
        ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
        ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
        ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))
        ax.legend(classes, loc=(1.04, 0))
    else:
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])

        plt.title(pca_title)
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
        plt.legend(classes, loc=(1.04, 0))

        
    #########################
    #
    # Determine which genes had the biggest influence on PC1
    #
    #########################

    ## get the name of the top 10 measurements (genes) that contribute
    ## most to pc1.
    ## first, get the loading scores
    if top_range_loadingfactors != None:
        loading_scores = pd.Series(pca.components_[0], index=embeddings_indecies)
        ## now sort the loading scores based on their magnitude
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

        # get the names of the top 10 genes
        top_loading_factors = sorted_loading_scores[top_range_loadingfactors[0]: top_range_loadingfactors[1]].index.values

        ## print the gene names and their scores (and +/- sign)
        print(loading_scores[top_loading_factors])

def get_centroids(embeddings, foreground, background):
    centroid_background = embeddings[background].sum(0, keepdims=True) / len(background)
    centroid_foreground = embeddings[foreground].sum(0, keepdims=True) / len(foreground)
    return [centroid_background, centroid_foreground]

def embeddings_accuracy(embeddings, targets, shape, centriods, num_classes=2):
    correct = 0
    wrong = 0
    heat_map = np.zeros(len(embeddings))
    for i in range(num_classes):
        inds = np.where(targets==i)[0]
        for emb_idx, point in enumerate(embeddings[inds]):
            min_dist = np.inf
            min_dist_class = -1
            for class_idx, c in enumerate(centriods):
                dist = np.linalg.norm(c-point)
                if(dist < min_dist):
                    # print("point: ({}, {}) which in class {}, closer to centroid: ({}, {}) which in class {}".format(x, y, i, c[0], c[1], idx))
                    min_dist_class = class_idx
                    min_dist = dist
                    heat_map[inds[emb_idx]] = class_idx
            if(i == min_dist_class):
                correct += 1
            else:
                wrong += 1
    print("correct: {}, wrong: {}, accuracy: {}%".format(correct, wrong, (100.0*correct)/(correct + wrong)))
    heat_map = heat_map.reshape(shape)
    return heat_map