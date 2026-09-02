from eval.unconstrained.models.stgcn import STGCN
import pandas as pd
import os.path as osp
import os
import datetime

import torch

from torch.utils.data import DataLoader
import numpy as np
import sys as _sys
from eval.a2m.action2motion.fid import calculate_fid
from eval.a2m.action2motion.diversity import calculate_diversity
from eval.unconstrained.metrics.kid import calculate_kid
from eval.unconstrained.metrics.precision_recall import precision_and_recall
from matplotlib import pyplot as plt

from eval.motion_pred import *
from data_loaders.grab.utils.config import Config
import pickle

TEST = False


def initialize_model(device, modelpath):
    num_classes = 12
    model = STGCN(in_channels=3,
                  num_class=num_classes,
                  graph_args={"layout": 'openpose', "strategy": "spatial"},
                  edge_importance_weighting=True,
                  device=device)
    model = model.to(device)
    state_dict = torch.load(modelpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def initialize_model_my(device, modelpath):
    cfg_classifier = Config("grab_classifier", test=False)
    model_classifier = get_action_classifier(cfg_classifier, 165, max_len=501)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    # scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model_classifier.parameters())) / 1000000.0))
    cp_path = cfg_classifier.vae_model_path % 500
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    model_classifier.load_state_dict(model_cp['model_dict'])
    model_classifier.to(device)
    model_classifier.eval()
    return model_classifier

def calculate_activation_statistics(activations):
    activations = activations.cpu().detach().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def compute_features(model, iterator, device):
    activations = []
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch_for_model = {}
            batch_for_model['x'] = batch.to(device).float()
            model(batch_for_model)
            activations.append(batch_for_model['features'])
            predictions.append(batch_for_model['yhat'])
            # labels.append(batch_for_model['y'])
        activations = torch.cat(activations, dim=0)
        predictions = torch.cat(predictions, dim=0)
    return activations, predictions


def evaluate_unconstrained_metrics(generated_motions, device, fast):

    act_rec_model_path = './assets/actionrecognition/humanact12_gru_modi_struct.pth.tar'
    dataset_path = './dataset/HumanAct12Poses/humanact12_unconstrained_modi_struct.npy'

    #act_rec_model_path = './assets/actionrecognition/vae_0500.p'
    #dataset_path = './dataset/grab_100_501_wact_candi_test.npz'

    # initialize model
    act_rec_model = initialize_model_my(device, act_rec_model_path)

    generated_motions -= generated_motions[:, 8:9, :, :]  # locate root joint of all frames at origin

    iterator_generated = DataLoader(generated_motions, batch_size=64, shuffle=False, num_workers=8)

    # compute features of generated motions
    generated_features, generated_predictions = compute_features(act_rec_model, iterator_generated, device=device)
    generated_stats = calculate_activation_statistics(generated_features)


    # dataset motions
    motion_data_raw = np.load(dataset_path, allow_pickle=True)
    motion_data = motion_data_raw[:, :15]  # data has 16 joints for back compitability with older formats
    motion_data -= motion_data[:, 8:9, :, :]  # locate root joint of all frames at origin
    iterator_dataset = DataLoader(motion_data, batch_size=64, shuffle=False, num_workers=8)

    # compute features of dataset motions
    dataset_features, dataset_predictions = compute_features(act_rec_model, iterator_dataset, device=device)
    real_stats = calculate_activation_statistics(dataset_features)

    print("evaluation resutls:\n")

    fid = calculate_fid(generated_stats, real_stats)
    print(f"FID score: {fid}\n")

    print("calculating KID...")
    kid = calculate_kid(dataset_features.cpu(), generated_features.cpu())
    (m, s) = kid
    print('KID : %.3f (%.3f)\n' % (m, s))

    dataset_diversity = calculate_diversity(dataset_features)
    generated_diversity = calculate_diversity(generated_features)
    print(f"Diversity of generated motions: {generated_diversity}")
    print(f"Diversity of dataset motions: {dataset_diversity}\n")

    if fast:
        print("Skipping precision-recall calculation\n")
        precision = recall = None
    else:
        print("calculating precision recall...")
        precision, recall = precision_and_recall(generated_features, dataset_features)
        print(f"precision: {precision}")
        print(f"recall: {recall}\n")

    metrics = {'fid': fid, 'kid': kid[0], 'diversity_gen': generated_diversity.cpu().item(), 'diversity_gt':  dataset_diversity.cpu().item(),
                 'precision': precision, 'recall':recall}
    return metrics

