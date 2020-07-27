import torch
import copy
import numpy as np
from tqdm import tqdm
import random
import time

def get_triplet(labels, foreground, background):
    anchor_label = random.randint(0, 1)
    if anchor_label == 0:
        # choose index from foreground and background
        anchor_idx = random.randrange(0, len(background))
        pos_idx = random.randrange(0, len(background))
        while anchor_idx == pos_idx:
            pos_idx = random.randrange(0, len(background))
        neg_idx = random.randrange(0, len(foreground))

        # get real index from outputs 
        anchor_idx = background[anchor_idx]
        pos_idx = background[pos_idx]
        neg_idx = foreground[neg_idx]

    elif anchor_label == 1:
        # choose index from foreground and background
        anchor_idx = random.randrange(0, len(foreground))
        pos_idx = random.randrange(0, len(foreground))
        while anchor_idx == pos_idx:
            pos_idx = random.randrange(0, len(foreground))
        neg_idx = random.randrange(0, len(background))

        # get real index from outputs
        anchor_idx = foreground[anchor_idx]
        pos_idx = foreground[pos_idx]
        neg_idx = background[neg_idx]
    return anchor_idx, pos_idx, neg_idx

def get_pixels_triplets(outputs, labels, batch_size, foreground, background):
    anchors = []
    positives = []
    negatives = []
    for _ in range(batch_size):
        anchor_idx, pos_idx, neg_idx = get_triplet(labels, foreground, background)            
        anchors.append(anchor_idx)
        positives.append(pos_idx)
        negatives.append(neg_idx)
    anchors = torch.tensor(anchors)
    positives =  torch.tensor(positives)
    negatives =  torch.tensor(negatives)
    return anchors, positives, negatives

def train_model(model, criterion, image, labels, optimizer, foreground, background, num_epochs=25, batch_size=16, num_batches=1, get_min_loss=True, use_gpu=True):
    if use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        image = image.to(device)
    model.train()
    min_run_loss = np.inf
    best_model_wts = None
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = []
        for _ in range(batch_size*num_batches):
            optimizer.zero_grad()
            outputs = model(image)
            anchor_idx, pos_idx, neg_idx = get_triplet(labels, foreground, background)
            loss = criterion(outputs, anchor_idx, pos_idx, neg_idx)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        mean_run_loss = np.mean(running_loss)
        print('loss: {:4f}'.format(mean_run_loss))
        if min_run_loss > mean_run_loss:
            min_run_loss = mean_run_loss
            if get_min_loss:
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val loss: {:4f}'.format(min_run_loss))
    
    if get_min_loss:
        model.load_state_dict(best_model_wts)
    return model

def train_model_with_mini_batch(model, criterion, image, labels, optimizer, foreground, background, num_epochs=25, batch_size=64, num_batches=1, get_min_loss=True, use_gpu=True):
    if use_gpu:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        image = image.to(device)
    model.train()
    min_run_loss = np.inf
    best_model_wts = None
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = []
        for _ in range(num_batches):
            optimizer.zero_grad()
            outputs = model(image)
            anchors, positives, negatives = get_pixels_triplets(outputs, labels, batch_size, foreground, background)
            loss = criterion(outputs, anchors, positives, negatives)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        mean_run_loss = np.mean(running_loss)
        print('loss: {:4f}'.format(mean_run_loss))
        if min_run_loss > mean_run_loss:
            min_run_loss = mean_run_loss
            if get_min_loss:
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val loss: {:4f}'.format(min_run_loss))
    
    if get_min_loss:
        model.load_state_dict(best_model_wts)
    return model

def get_pixels_optimized(outputs, labels, batch_size, anchors_positives, negatives):
    anchors = torch.tensor(random.sample(anchors_positives, batch_size))
    positives = torch.tensor(random.sample(anchors_positives, batch_size))
    negatives = torch.tensor(random.sample(negatives, batch_size))
    return anchors, positives, negatives

def train_cpu_optimized(model, criterion, image, labels, optimizer, foreground, background, num_epochs=25, batch_size=64, num_batches=1):
    since = time.clock()
    model.train()
    for epoch in range(num_epochs):
        running_loss = []
        for _ in range(num_batches):
            # foreground is the anchor
            optimizer.zero_grad()
            outputs = model(image)
            anchors, positives, negatives = get_pixels_triplets(outputs, labels, batch_size, foreground, background)
            loss = criterion(outputs, anchors, positives, negatives)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
            # background is the anchor
            optimizer.zero_grad()
            outputs = model(image)
            anchors, positives, negatives = get_pixels_optimized(outputs, labels, batch_size, background, foreground)
            loss = criterion(outputs, anchors, positives, negatives)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())

        mean_run_loss = np.mean(running_loss)
        print('loss: {:4f}'.format(mean_run_loss))
    now = time.clock()
    time_passed = now - since
    print('Time passed: {:}'.format(time_passed))
    # return model not necessary but to keep the same convention as the other training functions
    return model, time_passed