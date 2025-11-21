import argparse
import logging
import math
import os
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from models.discriminator import FCDiscriminator_img, grad_reverse
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, create_da_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, check_dataset_da, check_dataset_umt, \
    non_max_suppression, xyxy2xywhn, xywh2xyxy, xyxy2xywh, xywhn2xyxy, scale_coords
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA, ConsClsLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution, feature_visualization, plot_one_box
from utils.torch_utils import EarlyStopping, ModelEMA, WeightEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.ensemble_boxes_wbf import weighted_boxes_fusion
from utils.soft_nms import apply_soft_nms, apply_nms
import json
from torchvision.utils import save_image
import cv2
import pandas as pd

class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

logger = logging.getLogger(__name__)

@torch.no_grad()
def _update_teacher_model(student_model, teacher_model, keep_rate=0.9996):
    if opt.global_rank != -1:
        student_model_dict = {
            key[7:]: value for key, value in student_model.state_dict().items()
        }
    else:
        student_model_dict = student_model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (student_model_dict[key]*(1 - keep_rate) + value * keep_rate)
    
    teacher_model.load_state_dict(new_teacher_dict)

@torch.no_grad()
def _update_teacher_model_eman(student_model, teacher_model, keep_rate=0.9996):
    if opt.global_rank != -1:
        student_model_dict = {
            key[7:]: value for key, value in student_model.state_dict().items()
        }
    else:
        student_model_dict = student_model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            # new_teacher_dict[key] = (student_model_dict[key]*(1 - keep_rate) + value * keep_rate)
            if 'num_batches_tracked' in key:
                new_teacher_dict[key] = student_model_dict[key]
            else:
                new_teacher_dict[key] = (student_model_dict[key]*(1 - keep_rate) + value * keep_rate)
    
    teacher_model.load_state_dict(new_teacher_dict)


@torch.no_grad()
def momentum_update_ema(teacher_model, student_model, alpha=0.9996):
    teacher_named_parameters = dict(teacher_model.named_parameters())
    student_named_parameters = dict(student_model.named_parameters())
    
    # 교사 모델의 파라미터를 학생 모델의 파라미터로 업데이트
    for name, param in student_named_parameters.items():
        if name in teacher_named_parameters:
            teacher_param = teacher_named_parameters[name]
            # EMA 적용: teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
            teacher_param.data.copy_(alpha * teacher_param.data + (1 - alpha) * param.data)

            # oldvalue = teacher_param.data
            # value = alpha * teacher_param.data + (1 - alpha) * param.data
            # teacher_param.data.copy_(value)
            # changevalue = (value - oldvalue).norm()
            # logger.info('momentum_update_ema name[%s] changevalue[%f]' % (name, changevalue))

@torch.no_grad()
def momentum_update_eman(teacher_model, student_model, alpha=0.9996):
    state_dict_student = student_model.state_dict()
    state_dict_teacher = teacher_model.state_dict()
    for (k_main, v_main), (k_teacher, v_teacher) in zip(state_dict_student.items(), state_dict_teacher.items()):
        assert k_main == k_teacher, "state_dict names are different!"
        assert v_main.shape == v_teacher.shape, "state_dict shapes are different!"
        if 'num_batches_tracked' in k_teacher:
            v_teacher.copy_(v_main)
        else:
            v_teacher.copy_(v_teacher * alpha + (1. - alpha) * v_main)

def model_deepcopy(model):
    new_model = type(model)(showinfo=False).to(device)
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), new_model.parameters()):
            new_param.data.copy_(param.data)
        for buffer, new_buffer in zip(model.buffers(), new_model.buffers()):
            new_buffer.data.copy_(buffer.data)

    new_model.eval()
    return new_model


def get_predict_list(prediction, conf_thres, w=640, h=640):
    """
    return bboxes, scores, bbclasses
    """
    xc = prediction[..., 4] > conf_thres  # candidates

    bboxes = [torch.zeros((0, 4), device=prediction.device)] * prediction.shape[0]
    scores = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    bbclasses = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        #box = xywhn2xyxy(x[:, :4], w=w, h=w)
        conf, j = x[:, 5:].max(1, keepdim=True)
        #x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        x = x[conf.view(-1) > conf_thres]
        box = xywh2xyxy(x[:, :4])
        _, j = x[:, 5:].max(1, keepdim=True)
        bboxes[xi] = box
        scores[xi] = x[:,4].unsqueeze(1)
        bbclasses[xi] = j

    return bboxes, scores, bbclasses

def adjust_weights(epoch, alpha_start, alpha_end, beta_start, beta_end, num_epochs):
    t = epoch / num_epochs
    alpha = (1 - t) * alpha_start + t * alpha_end
    beta = (1 - t) * beta_start + t * beta_end
    return alpha, beta

def adjust_weights_exponential(epoch, alpha_start, alpha_end, beta_start, beta_end, num_epochs):
    # Calculate the decay factor
    alpha_decay = (alpha_end / alpha_start) ** (1 / num_epochs)
    beta_decay = (beta_end / beta_start) ** (1 / num_epochs)
    
    # Calculate the adjusted weights
    alpha = alpha_start * (alpha_decay ** epoch)
    beta = beta_start * (beta_decay ** epoch)
    
    return alpha, beta

def fusion_weak_strong(bboxes, scores, clses, bboxes_s, scores_s, clses_s, per_batch_size, device, iou_thres, conf_thres, wbf_iou_thr):
    bboxlist, scorelist, clslist = [],[],[]
    #====================> Fuse and merge w/s one by one
    #"""
    pred_labels_out_batch = [] 
    for img_id in range(per_batch_size):
        bboxlist.append(bboxes[img_id])
        #bboxlist.append(bboxes_s[img_id])
        scorelist.append(scores[img_id])
        #scorelist.append(scores_s[img_id])
        clslist.append(clses[img_id])
        #clslist.append(clses_s[img_id])

        boxes_1, scores_1, labels_1 = weighted_boxes_fusion(bboxlist, scorelist, clslist, iou_thr=wbf_iou_thr, conf_type="avg", skip_box_thr=0.2)
        #boxes_1, scores_1, labels_1 = weighted_boxes_fusion(bboxlist, scorelist, clslist, iou_thr=iou_thres, conf_type="avg", skip_box_thr=0.2)
        boxes_1 = torch.tensor(boxes_1)
        scores_1 = torch.tensor(scores_1)
        labels_1 = torch.tensor(labels_1)

        bboxlist.clear()
        scorelist.clear()
        clslist.clear()

        bboxlist.append(bboxes_s[img_id])
        scorelist.append(scores_s[img_id])
        clslist.append(clses_s[img_id])

        boxes_2, scores_2, labels_2 = weighted_boxes_fusion(bboxlist, scorelist, clslist, iou_thr=wbf_iou_thr, conf_type="avg", skip_box_thr=0.2)
        #boxes_2, scores_2, labels_2 = weighted_boxes_fusion(bboxlist, scorelist, clslist, iou_thr=iou_thres, conf_type="avg", skip_box_thr=0.2)
        boxes_2 = torch.tensor(boxes_2)
        scores_2 = torch.tensor(scores_2)
        labels_2 = torch.tensor(labels_2)

        bboxlist.clear()
        scorelist.clear()
        clslist.clear()

        # WBF-WBF ---------------------------
        # bboxlist.append(boxes_1)
        # scorelist.append(scores_1)
        # clslist.append(labels_1)
        # bboxlist.append(boxes_2)
        # scorelist.append(scores_2)
        # clslist.append(labels_2)
        
        # boxes_, scores_, labels_ = weighted_boxes_fusion(bboxlist, scorelist, clslist, iou_thr=iou_thres, conf_type="box_and_model_avg", skip_box_thr=0.25)
        # boxes_ = torch.tensor(boxes_)
        # scores_ = torch.tensor(scores_)
        # labels_ = torch.tensor(labels_)
        # WBF-WBF ---------------------------

        wbf_boxes = np.concatenate((boxes_1, boxes_2), axis=0)
        wbf_scores = np.concatenate((scores_1, scores_2), axis=0)
        wbf_labels = np.concatenate((labels_1, labels_2), axis=0)

        # WBF-SOFT---------------------------
        boxes_, scores_, labels_ = apply_soft_nms(wbf_boxes, wbf_scores, wbf_labels, sigma=0.5, score_threshold=conf_thres, iou_threshold=iou_thres, method='gaussian')
        
        # WBF-NMS---------------------------
        # boxes_, scores_, labels_ = apply_nms(wbf_boxes, wbf_scores, wbf_labels, iou_thres)

        labels_num = labels_.shape[0]
        labels_tensor = torch.tensor(labels_).to(device)
        boxes_tensor = torch.tensor(boxes_).to(device)
        labels_list = torch.cat((labels_tensor.unsqueeze(1), boxes_tensor), dim=1).to(device)
        #labels_list = torch.cat((labels_.unsqueeze(1), boxes_), dim=1).to(device)
        labels_list[:, 1:5] = xyxy2xywh(labels_list[:, 1:5])  # xyxy to xywh
        pred_labels_out = torch.cat(((torch.ones(labels_num)*img_id).unsqueeze(-1).to(device), 
                labels_list), dim=1)  # pred_labels_out shape is (labels_num, 6), per label format [img_id cls x y x y]
        pred_labels_out_batch.append(pred_labels_out)
    #"""
    #====================< 
    
    if len(pred_labels_out_batch) != 0:
        pred_labels = torch.cat(pred_labels_out_batch, dim=0)
    else:
        # pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, 1, 1]])).to(device)
        pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, random.uniform(0.2,0.8), random.uniform(0.2,0.8)]])).to(device)
    
    return pred_labels



def fusion(bboxes, scores, clses, per_batch_size, device, iou_thres):
    bboxlist, scorelist, clslist = [],[],[]
    #====================> w/s 하나씩 fusion 해서 합침
    pred_labels_out_batch = [] 
    for img_id in range(per_batch_size):
        bboxlist.append(bboxes[img_id])
        #bboxlist.append(bboxes_s[img_id])
        scorelist.append(scores[img_id])
        #scorelist.append(scores_s[img_id])
        clslist.append(clses[img_id])
        #clslist.append(clses_s[img_id])

        boxes_, scores_, labels_ = weighted_boxes_fusion(bboxlist, scorelist, clslist, iou_thr=iou_thres, conf_type="avg", skip_box_thr=0.2)
        boxes_ = torch.tensor(boxes_)
        scores_ = torch.tensor(scores_)
        labels_ = torch.tensor(labels_)

        bboxlist.clear()
        scorelist.clear()
        clslist.clear()

        labels_num = labels_.shape[0]
        labels_tensor = torch.tensor(labels_).to(device)
        boxes_tensor = torch.tensor(boxes_).to(device)
        labels_list = torch.cat((labels_tensor.unsqueeze(1), boxes_tensor), dim=1).to(device)
        #labels_list = torch.cat((labels_.unsqueeze(1), boxes_), dim=1).to(device)
        labels_list[:, 1:5] = xyxy2xywh(labels_list[:, 1:5])  # xyxy to xywh
        pred_labels_out = torch.cat(((torch.ones(labels_num)*img_id).unsqueeze(-1).to(device), 
                labels_list), dim=1)  # pred_labels_out shape is (labels_num, 6), per label format [img_id cls x y x y]
        pred_labels_out_batch.append(pred_labels_out)
    #"""
    #====================< w/s 하나씩 fusion 해서 합침
        
    if len(pred_labels_out_batch) != 0:
        pred_labels = torch.cat(pred_labels_out_batch, dim=0)
    else:
        # pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, 1, 1]])).to(device)
        pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, random.uniform(0.2,0.8), random.uniform(0.2,0.8)]])).to(device)
    
    return pred_labels


def fusion_all(bboxes, scores, clses, bboxes_s, scores_s, clses_s, s_weight, per_batch_size, device, iou_thres):

    bboxlist, scorelist, clslist, weights = [],[],[],[]
    #====================>  w/s 한번에 넣고 fusion
    
    pred_labels_out_batch = [] 
    for img_id in range(per_batch_size):
        bboxlist.append(bboxes[img_id])
        weights.append(1)
        bboxlist.append(bboxes_s[img_id])
        weights.append(s_weight)
        scorelist.append(scores[img_id])
        scorelist.append(scores_s[img_id])
        clslist.append(clses[img_id])
        clslist.append(clses_s[img_id])

        boxes_, scores_, lables_ = weighted_boxes_fusion(bboxlist, scorelist, clslist, weights=weights, iou_thr=iou_thres, conf_type="box_and_model_avg", skip_box_thr=0.25)
        boxes_ = torch.tensor(boxes_)
        scores_ = torch.tensor(scores_)
        lables_ = torch.tensor(lables_)

        labels_num = lables_.size()
        labels_list = torch.cat((lables_.unsqueeze(1), boxes_), dim=1).to(device)
        labels_list[:, 1:5] = xyxy2xywh(labels_list[:, 1:5])  # xyxy to xywh
        pred_labels_out = torch.cat(((torch.ones(labels_num)*img_id).unsqueeze(-1).to(device), 
                labels_list), dim=1)  # pred_labels_out shape is (labels_num, 6), per label format [img_id cls x y x y]
        pred_labels_out_batch.append(pred_labels_out)

        bboxlist.clear()
        scorelist.clear()
        clslist.clear()
        weights.clear()
    #====================<  w/s 한번에 넣고 fusion


    if len(pred_labels_out_batch) != 0:
        pred_labels = torch.cat(pred_labels_out_batch, dim=0)
    else:
        # pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, 1, 1]])).to(device)
        pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, random.uniform(0.2,0.8), random.uniform(0.2,0.8)]])).to(device)

    return pred_labels


def check_moving_avg_and_gradient(loss_history, window_size, epoch, opt):

    gradient_threshold =  opt.gradient_thr
    moving_avg_diff_threshold = opt.moving_avg_diff_thr

    for loss_idx, history in loss_history.items():
        if len(history) >= window_size:
            # 이동 평균 계산
            moving_avg = pd.Series(history).rolling(window=window_size).mean().iloc[-1]
            prev_moving_avg = pd.Series(history).rolling(window=window_size).mean().iloc[-2]
            
            # 변화율(Gradient) 계산
            smoothed_history = pd.Series(history).rolling(window=window_size).mean().dropna()
            if len(smoothed_history) < 2:
                gradients = np.zeros(len(history))
            else:
                gradients = np.gradient(smoothed_history)

            current_gradient = gradients[-1]

            print(f"loss_idx: {loss_idx}, moving_avg: {moving_avg}, prev_moving_avg: {prev_moving_avg}, current_gradient: {current_gradient},")

            # 조건: 이동 평균 상승 & 변화율 양수
            find = False
            update = True
            if (moving_avg - prev_moving_avg) > moving_avg_diff_threshold and current_gradient > gradient_threshold:
                find = True
                update = False
            elif (prev_moving_avg - moving_avg) > moving_avg_diff_threshold and current_gradient < -gradient_threshold:
                find = True
                update = True

            value_dict = {'update':[], 'item': [], 'moving_avg': [], 'prev_moving_avg':[],  'gradient': [], 'update_stop_iter': []}
            value_dict['update'].append(update)
            value_dict['item'].append(loss_idx)
            value_dict['moving_avg'].append(moving_avg)
            value_dict['prev_moving_avg'].append(prev_moving_avg)
            value_dict['gradient'].append(current_gradient)
            value_dict['update_stop_iter'].append(epoch)

            df = pd.DataFrame(value_dict)
            file_path = f'./check_moving_avg_and_gradient_{opt.name}.csv'
            if not os.path.isfile(file_path):
                df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, mode='a', index=False, header=False)

            if find:
                return find, update, loss_idx, moving_avg, prev_moving_avg, current_gradient           

    return False, True, 0, 0, 0, 0

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # add teacher, student
    teacher_alpha, conf_thres, iou_thres, max_gt_boxes, lambda_weight, student_weight, teacher_weight = \
        opt.teacher_alpha, opt.conf_thres, opt.iou_thres, opt.max_gt_boxes, opt.lambda_weight, \
        opt.student_weight, opt.teacher_weight

    # torch.autograd.set_detect_anomaly(True)

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # last = wdir / 'last.pt'
    # best = wdir / 'best.pt'
    last_student, last_teacher = wdir / 'last_student.pt', wdir / 'last_teacher.pt'
    best_student, best_teacher = wdir / 'best_student.pt', wdir / 'best_teacher.pt'

    results_file = save_dir / 'results.txt'
    results_file_t = save_dir / 'results_teacher.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        # with torch_distributed_zero_first(rank):
        #     attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        #model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_student = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_teacher = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        domain_img = FCDiscriminator_img(num_classes=1024).to(device)

        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        if 'domain' in ckpt:
            domain_dict = ckpt['domain'].float().state_dict()
            domain_img.load_state_dict(domain_dict, strict=False) 
                
        state_dict = intersect_dicts(state_dict, model_student.state_dict(), exclude=exclude)  # intersect
        model_student.load_state_dict(state_dict, strict=False)  # load
        model_teacher.load_state_dict(state_dict.copy(), strict=False)  # load

        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model_student.state_dict()), weights))  # report

    else:
        #model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_student = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_teacher = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        domain_img = FCDiscriminator_img(num_classes=1024).to(device)

    for p in domain_img.parameters():
        p.requires_grad = True

    # Update models weights [only by this way, we can resume the old training normally...][ref models.experimental.attempt_load()]
    if student_weight != "None" and teacher_weight != "None":  # update model_student and model_teacher
        torch.cuda.empty_cache()
        ckpt_student = torch.load(student_weight, map_location=device)  # load checkpoint
        state_dict_student = ckpt_student['ema' if ckpt_student.get('ema') else 'model'].float().half().state_dict()  # to FP32
        model_student.load_state_dict(state_dict_student, strict=False)  # load
        del ckpt_student, state_dict_student
        
        ckpt_teacher = torch.load(teacher_weight, map_location=device)  # load checkpoint
        state_dict_teacher = ckpt_teacher['ema' if ckpt_teacher.get('ema') else 'model'].float().half().state_dict()  # to FP32
        model_teacher.load_state_dict(state_dict_teacher, strict=False)  # load
        del ckpt_teacher, state_dict_teacher

    with torch_distributed_zero_first(rank):
        check_dataset_da(data_dict)
        #check_dataset(data_dict)  # check
        #check_dataset_umt(data_dict)
    
    train_source_path = data_dict['source']
    train_path = data_dict['train']
    train_unlabel_path = data_dict['train_unlabel']
    test_path = data_dict['test']
    val_path = data_dict['val']
        
    # Freeze
    freeze_student = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model_student.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze_student):
            print('freezing %s' % k)
            v.requires_grad = False

    #freeze_teacher = []  # parameter names to freeze (full or partial)
    freeze_teacher = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model_teacher.named_parameters():
        v.requires_grad = False #True  # train all layers
        if any(x in k for x in freeze_teacher):
            print('freezing %s' % k)
            v.requires_grad = False
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model_student.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        student_optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        optim_domain = optim.Adam(domain_img.parameters(), lr=2e-4, betas=(0.9, 0.999)) 
    else:
        student_optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optim_domain = optim.SGD(domain_img.parameters(), lr=3e-3, weight_decay=1e-4, momentum=hyp['momentum'], nesterov=True)

    student_optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    student_optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2


    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(student_optimizer, lr_lambda=lf)
    #scheduler = WarmupConstantSchedule(student_optimizer, warmup_steps=5)

    # EMA
    ema = ModelEMA(model_student) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness, best_fitness_t = 0, 0.0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            student_optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model_student.stride.max()), 32)  # grid size (max stride)
    nl = model_student.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
        model_teacher = torch.nn.DataParallel(model_teacher)
        model_student = torch.nn.DataParallel(model_student)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model_student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_student).to(device)
        model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader_t, dataset_t = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    dataloader_t_un, dataset_t_un = create_da_dataloader(train_unlabel_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train_unlabel: '))
    

    mlc = np.concatenate(dataset_t.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader_t_un)
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader, dataset_test = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))

        if not opt.resume:
            labels = np.concatenate(dataset_test.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                # check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                check_anchors(dataset_t, model=model_student, thr=hyp['anchor_t'], imgsz=imgsz)
                check_anchors(dataset_t, model=model_teacher, thr=hyp['anchor_t'], imgsz=imgsz)

            # model.half().float()  # pre-reduce anchor precision
            model_student.half().float()  # pre-reduce anchor precision
            model_teacher.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model_student = DDP(model_student, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model_student.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model_student.nc = nc  # attach number of classes to model
    model_student.hyp = hyp  # attach hyperparameters to model
    model_student.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model_student.class_weights = labels_to_class_weights(dataset_t.labels, nc).to(device) * nc  # attach class weights
    model_student.names = names
    model_teacher.nc = nc  # attach number of classes to model
    model_teacher.hyp = hyp  # attach hyperparameters to model
    model_teacher.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model_teacher.class_weights = labels_to_class_weights(dataset_t.labels, nc).to(device) * nc  # attach class weights
    model_teacher.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    
    results = (0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.75, mAP@.5-.95, val_loss(box, obj, cls)  # Added in 2021-10-01
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model_student)  # init loss class
    compute_loss = ComputeLoss(model_student)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader_t_un.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    if opt.teacher_earlystopping:
        stopper, stop = EarlyStopping(patience=opt.patience), False


    # 초기 가중치 설정
    alpha_start = opt.alpha_start  # 초기 Supervised Loss 가중치
    alpha_end = opt.alpha_end  # 최종 Supervised Loss 가중치
    beta_start = opt.beta_start  # 초기 Unsup Loss 가중치
    beta_end = opt.beta_end  # 최종 Unsup Loss 가중치

    #loss_history = []
    loss_history = {i: [] for i in range(5)}  # 각 loss별 기록
    window_size = opt.window_size
   
    teacher_update = True
    res_dict = {'update':[], 'item': [], 'moving_avg': [], 'prev_moving_avg':[],  'gradient': [], 'update_stop_iter': []}
    res_avg_dict = {'update': [], 'item': [], 'avg_gradient':[],  'update_stop_iter': []}

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model_student.train()
        model_teacher.train()

        #alpha, beta = adjust_weights(epoch, alpha_start, alpha_end, beta_start, beta_end, epochs)
        alpha, beta = adjust_weights_exponential(epoch, alpha_start, alpha_end, beta_start, beta_end, epochs)

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model_student.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset_t.labels, nc=nc, class_weights=cw)  # image weights
                dataset_t.indices = random.choices(range(dataset_t.n), weights=iw, k=dataset_t.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset_t.indices) if rank == 0 else torch.zeros(dataset_t.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset_t.indices = indices.cpu().numpy()
                    
        addlogcount = 1 # domain
        if opt.distill_loss:
            addlogcount += 1

        if opt.consistency_loss:
            addlogcount += 1

        if opt.contrastive:
            addlogcount += 1
        
        mloss = torch.zeros(4 + addlogcount, device=device)  # mean losses

        if rank != -1:
            #dataloader_s.sampler.set_epoch(epoch)
            dataloader_t.sampler.set_epoch(epoch)
            dataloader_t_un.sampler.set_epoch(epoch)
            #dataloader.sampler.set_epoch(epoch)

        #pbar = enumerate(dataloader)
        pbar = enumerate([ind for ind in range(nb)])
        #data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        data_iter_t_un = iter(dataloader_t_un)

        #logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if addlogcount > 0:
            #log_list = ['Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'dom_s', 'dom_t', 'labels', 'img_size']
            log_list = ['Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'dom_t', 'labels', 'img_size']
            pos = 6
            shift = 0

            if opt.distill_loss:
                log_list = log_list[:pos] + ['distill'] + log_list[pos:]
                pos += 1
            shift += opt.distill_loss

            if opt.consistency_loss: 
                log_list = log_list[:pos] + ['cons'] + log_list[pos:]
                pos += 1
            shift += opt.consistency_loss
            
            if opt.contrastive:
                log_list = log_list[:pos] + ['cont'] + log_list[pos:]
            shift += opt.contrastive


            logger.info(('\n' + '%10s' * (9 + shift)) % tuple(log_list))
        else:
            logger.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'dom_t', 'labels', 'img_size'))


        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        student_optimizer.zero_grad()
        optim_domain.zero_grad()
        
        if isinstance(model_student, torch.nn.parallel.DistributedDataParallel):
            if model_student.module.backbone_features.grad is not None: 
                model_student.module.backbone_features.grad.zero_()
        else:
            #model_student.backbone_features.zero_grad()
            if model_student.backbone_features.grad is not None: 
                model_student.backbone_features.grad.zero_()

        loss_historyitem = 0

        #for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        for i, ind in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            #imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            try:
                #imgs_s_w, sources_w, imgs_s_s, sources_s, paths_s, _ = next(data_iter_s)
                imgs_t, targets_t, paths_t, _ = next(data_iter_t)
                imgs_t_un_w, targets_un_w, imgs_t_un_s, targets_un_s, paths_t_un, _ = next(data_iter_t_un)
            except:
                data_iter_t = iter(dataloader_t)
                data_iter_t_un = iter(dataloader_t_un)
                imgs_t, targets_t, paths_t, _ = next(data_iter_t)
                imgs_t_un_w, targets_un_w, imgs_t_un_s, targets_un_s, paths_t_un, _ = next(data_iter_t_un)
            imgs_t = imgs_t.to(device, non_blocking=True).float() / 255.0
            imgs_t_un_w = imgs_t_un_w.to(device, non_blocking=True).float() / 255.0
            imgs_t_un_s = imgs_t_un_s.to(device, non_blocking=True).float() / 255.0
            targets_t = targets_t.to(device)
            targets_un_w = targets_un_w.to(device)
            targets_un_s = targets_un_s.to(device)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(student_optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                temp_imgs_list = [imgs_t] #[imgs_t_w, imgs_t_s]
                for i, imgs in enumerate(temp_imgs_list):
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    temp_imgs_list[i] = imgs

                [imgs_t] = temp_imgs_list
                #[imgs_t_w, imgs_t_s] = temp_imgs_list

            if ni % accumulate == 0:
                optim_domain.zero_grad(set_to_none=True)
                student_optimizer.zero_grad(set_to_none=True)

            for p in domain_img.parameters():
                p.requires_grad = False

            # Forward
            with amp.autocast(enabled=cuda):
                # pred = model(imgs)  # forward
                # if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                #     loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                # else:
                #     loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                model_student.zero_grad()
                model_teacher.eval()

                per_batch_size, channels, height, width = imgs_t_un_w.shape



                # make peuso label : target unlabel week만 사용
                imgs_t_un_s_flip = torch.flip(imgs_t_un_s, dims=[3])

                predt_tw_inf, predt_tw = model_teacher(imgs_t_un_w)  # forward. when eval(), the output is (x1, x2) in yolo.py
                predt_ts_inf, predt_ts = model_teacher(imgs_t_un_s_flip)  # forward. when eval(), the output is (x1, x2) in yolo.py

                # bboxes, scores, clses = get_predict_list(predt_tw_inf, conf_thres=0.25, w=width, h=height)
                bboxes, scores, clses = get_predict_list(predt_tw_inf, conf_thres=opt.pseudo_thres, w=width, h=height)
                bboxes = [x/opt.img_size[0] for x in bboxes]

                # bboxes_s, scores_s, clses_s = get_predict_list(predt_ts_inf, conf_thres=0.25, w=width, h=height)
                bboxes_s, scores_s, clses_s = get_predict_list(predt_ts_inf, conf_thres=opt.pseudo_thres, w=width, h=height)
                bboxes_s = [x/opt.img_size[0] for x in bboxes_s]

                if opt.wbf_weight > 0:
                    pred_labels = fusion_all(bboxes, scores, clses, bboxes_s, scores_s, clses_s, s_weight=opt.wbf_weight, per_batch_size=per_batch_size, device=device, iou_thres=iou_thres)
                else:
                    pred_labels = fusion_weak_strong(bboxes, scores, clses, bboxes_s, scores_s, clses_s, per_batch_size, device, iou_thres=iou_thres, conf_thres=conf_thres, wbf_iou_thr=opt.wbf_iou_thres)

            
                # 정답지로 student 학습
                preds_ts = model_student(imgs_t)
                loss_sup, loss_items_sup = compute_loss(preds_ts, targets_t.detach().clone()) 


                source_label = 0
                target_label = 1

                # unlabel weak 
                preds_t_un_w = model_student(imgs_t_un_w)       # unlabel w

                # unlabel strong
                preds_t_un_s = model_student(imgs_t_un_s)
                if hasattr(model_student, 'module'):
                    features = model_student.module.get_backbone_features()
                else:
                    features = model_student.get_backbone_features()


                loss = loss_sup * alpha
                loss_items = loss_items_sup

                if opt.distill_loss:
                    # peudo supervised(target unlabel weak) 
                    loss_s_t_un_s, loss_items_s_t_un_s = compute_loss(preds_t_un_w, pred_labels.detach().clone()) 
                    loss_distill = loss_s_t_un_s * opt.lambda_weight
                    loss = loss + loss_distill*(beta/2)
                    loss_items[3] = loss_items[3] + loss_distill.detach()  # total

                    loss_items = loss_items + loss_items_s_t_un_s
                    loss_items = torch.cat((loss_items, loss_distill.detach()), 0)
                    
                if opt.use_tuc:
                    loss_historyitem += loss_items.detach()

                if opt.consistency_loss:
                    # s student와 w student의 KLDivLoss를 사용하여 예측값 비교하여 계산
                    #loss_cons = KD_loss(pred_tr_s[0][:,:,:,:,5:], pred_tr_t[0][:,:,:,:,5:])/(120*120*3)

                    # student w 예측값과 s 예측값의 차이
                    loss_cons = 0
                    loss_cons_cls, loss_cons_loc = 0, 0
                    bs = preds_t_un_w[0].shape[0]
                    reshaped_tensor_s = preds_t_un_s[0].clone().reshape(bs, -1, nc+5)
                    reshaped_tensor_w = preds_t_un_w[0].clone().reshape(bs, -1, nc+5)

                    loss_cons_cls_w2s, loss_cons_cls_s2w = 0, 0

                    cons_loc_loss_x = 0
                    cons_loc_loss_y = 0
                    cons_loc_loss_w = 0
                    cons_loc_loss_h = 0

                    for i in range(bs):
                        loss_cons_cls_w2s += ConsClsLoss(reshaped_tensor_w[i][:,5:].detach(), reshaped_tensor_s[i][:,5:])
                        loss_cons_cls_s2w += ConsClsLoss(reshaped_tensor_s[i][:,5:].detach(), reshaped_tensor_w[i][:,5:])
                        # x: (x_s + x_w - 1)**2
                        cons_loc_loss_x += torch.mean((reshaped_tensor_s[i][:, 0] + reshaped_tensor_w[i][:, 0] - 1)**2)
                        # y: (y_s - y_w)**2
                        cons_loc_loss_y += torch.mean((reshaped_tensor_s[i][:, 1] - reshaped_tensor_w[i][:, 1])**2)
                        # w: (w_s - w_w)**2
                        cons_loc_loss_w += torch.mean((reshaped_tensor_s[i][:, 2] - reshaped_tensor_w[i][:, 2])**2)
                        # h: (h_s - h_w)**2
                        cons_loc_loss_h += torch.mean((reshaped_tensor_s[i][:, 3] - reshaped_tensor_w[i][:, 3])**2)

                    loss_cons_cls_w2s /= bs
                    loss_cons_cls_s2w /= bs
                    cons_loc_loss_x /= bs
                    cons_loc_loss_y /= bs
                    cons_loc_loss_w /= bs
                    cons_loc_loss_h /= bs

                    loss_cons_cls = (loss_cons_cls_w2s + loss_cons_cls_s2w)/2
                    loss_cons_loc = (cons_loc_loss_x + cons_loc_loss_y + cons_loc_loss_w + cons_loc_loss_h) / 4

                    loss_cons_cls *= hyp['cls']
                    loss_cons_loc *= hyp['box']
                    loss_cons = loss_cons_cls + loss_cons_loc
                    #loss_cons = torch.div(loss_cons_cls, 2) + loss_cons_loc
                    loss_cons *= opt.alpha_weight

                    
                    # loss_fn = torch.nn.L1Loss()
                    # loss_cons = loss_fn(loss_tf_s, loss_tr_s) * opt.alpha_weight
                    loss = loss + loss_cons*(beta/2)
                    loss_items[3] = loss_items[3] + loss_cons.detach()  # total
                    loss_items = torch.cat((loss_items, loss_cons.detach().unsqueeze(0)), 0)

    
                features_s = grad_reverse(features)
                d_s = domain_img(features_s)
                loss_domain = F.binary_cross_entropy_with_logits(d_s, torch.FloatTensor(d_s.data.size()).fill_(target_label).to(device))

                loss = loss + loss_domain
                loss_items = torch.cat((loss_items, loss_domain.detach().unsqueeze(0)), 0)

                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            for p in domain_img.parameters():
                p.requires_grad = True

            with amp.autocast(enabled=cuda):                            
                feats_detached = features.detach()
                logit_D = domain_img(feats_detached)
                loss_D = F.binary_cross_entropy_with_logits(logit_D, torch.FloatTensor(logit_D.data.size()).fill_(target_label).to(device))
            
            scaler.scale(loss_D).backward()


            # Optimize
            if ni % accumulate == 0:
                scaler.step(optim_domain)
                scaler.step(student_optimizer)  
                optim_domain.zero_grad(set_to_none=True)
                student_optimizer.zero_grad(set_to_none=True)

                scaler.update()
                                
                if ema:
                    ema.update(model_student)

                if teacher_update:
                    if opt.use_eman:
                        momentum_update_eman(teacher_model= model_teacher, student_model=model_student, alpha=opt.teacher_alpha)
                    else:
                        momentum_update_ema(teacher_model= model_teacher, student_model=model_student, alpha=opt.teacher_alpha)
                    #_update_teacher_model(student_model=model_student, teacher_model=model_teacher, keep_rate=opt.teacher_alpha)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * (4 + addlogcount + 2)) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets_t.shape[0], imgs_t.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_t_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_t, targets_t, paths_t, f), daemon=True).start()
                    f = save_dir / f'train_t_un_w_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_t_un_w, targets_un_w, paths_t_un, f), daemon=True).start()
                    f = save_dir / f'train_t_un_s_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_t_un_s, targets_un_s, paths_t_un, f), daemon=True).start()

                    if ni == 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            tb_writer.add_graph(torch.jit.trace(de_parallel(model_student), imgs_t[0:1], strict=False), [])

                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 3 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in student_optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model_student, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                logger.info('epoch %d ' + ('%20s' + '%12s' * 6), epoch, 'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric,
                                                 logger=logger)

                results_t, maps_t, times_t = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=model_teacher,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric,
                                                 logger=logger)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
                #f.write(s + '%10.4g' * 8 % results + '\n')
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            with open(results_file_t, 'a') as f:
                f.write(s + '%10.4g' * 7 % results_t + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file_t, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/total_loss', 'domain_t', # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            
            tags_t = ['teacher_metrics/precision', 'teacher_metrics/recall', 'teacher_metrics/mAP_0.5', 'teacher_metrics/mAP_0.5:0.95',
                    'teacher_val/box_loss', 'teacher_val/obj_loss', 'teacher_val/cls_loss']
            
            tagpos = 4

            if opt.distill_loss:
                tags = tags[:tagpos] + ['train/distill_loss'] + tags[tagpos:]
                tagpos += 1

            if opt.consistency_loss: 
                tags = tags[:tagpos] + ['train/cons_loss'] + tags[tagpos:]
                tagpos += 1
            
            if opt.contrastive:
                tags = tags[:tagpos] + ['train/contrastive_loss'] + tags[tagpos:]
                tagpos += 1
            
            #for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            for x, tag in zip(list(mloss[:]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            for x, tag in zip(list(results_t), tags_t):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            fi_t = fitness(np.array(results_t).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if opt.teacher_earlystopping:
                stop = stopper(epoch=epoch, fitness=fi_t)  # early stop check
                if stop:
                    teacher_update = False

            if fi_t > best_fitness_t:
                best_fitness_t = fi_t
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                # ckpt = {'epoch': epoch,
                #         'best_fitness': best_fitness,
                #         'training_results': results_file.read_text(),
                #         'model': deepcopy(model.module if is_parallel(model) else model).half(),
                #         'ema': deepcopy(ema.ema).half(),
                #         'updates': ema.updates,
                #         'optimizer': optimizer.state_dict(),
                #         'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}
                ckpt_student = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model_student)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': student_optimizer.state_dict(),
                        'domain':deepcopy(domain_img),
                        'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}
                        
                ckpt_teacher = {
                        'epoch': epoch,
                        'best_fitness': best_fitness_t,
                        'model': deepcopy(de_parallel(model_teacher)).half(),
                        'domain':deepcopy(domain_img),
                        'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}
                # Save last, best and delete
                #torch.save(ckpt, last)
                torch.save(ckpt_student, last_student)
                torch.save(ckpt_teacher, last_teacher)
                
                if best_fitness == fi:
                    torch.save(ckpt_student, best_student)

                if best_fitness_t == fi_t:
                    torch.save(ckpt_teacher, best_teacher)

                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt_student, wdir / 'best_student_{:03d}.pt'.format(epoch))
                if (best_fitness_t == fi_t) and (epoch >= 200):
                    torch.save(ckpt_teacher, wdir / 'best_teacher_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt_student, wdir / 'epoch_student_{:03d}.pt'.format(epoch))
                    torch.save(ckpt_teacher, wdir / 'epoch_teacher_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt_student, wdir / 'epoch_student_{:03d}.pt'.format(epoch))
                    torch.save(ckpt_teacher, wdir / 'epoch_teacher_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-1):
                    torch.save(ckpt_student, wdir / 'epoch_student_{:03d}.pt'.format(epoch))
                    torch.save(ckpt_teacher, wdir / 'epoch_teacher_{:03d}.pt'.format(epoch))
                # if wandb_logger.wandb:
                #     if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                #         wandb_logger.log_model(
                #             last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                #del ckpt
                del ckpt_student, ckpt_teacher

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
        
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir, distill=opt.distill_loss, cons=opt.consistency_loss, cont=opt.contrastive, domain=True)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last_student, best_student) if best_student.exists() else (last_student):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric,
                                          logger=logger)

        # Strip optimizers
        final = best_student if best_student.exists() else last_student  # final model
        for f in last_student, best_student:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    parser.add_argument('--teacher_alpha', type=float, default=0.9996, help='Teacher EMA alpha (decay) in UMT')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='Confidence threshold for pseudo label in UMT')
    parser.add_argument('--iou_thres', type=float, default=0.3, help='Overlap threshold used for non-maximum suppression in UMT')
    parser.add_argument('--max_gt_boxes', type=int, default=20, help='Maximal number of gt rois in an image during training in UMT')
    parser.add_argument('--lambda_weight', type=float, default=0.01, help='The weight for distillation loss in UMT')
    #parser.add_argument('--lambda_weight', type=float, default=0.005, help='The weight for distillation loss in UMT')
    parser.add_argument('--pseudo_thres', type=float, default=0.75)
    parser.add_argument('--wbf_iou_thres', type=float, default=0.3)
    parser.add_argument('--alpha_start', type=float, default=0.9)
    parser.add_argument('--alpha_end', type=float, default=0.5)
    parser.add_argument('--beta_start', type=float, default=0.1)
    parser.add_argument('--beta_end', type=float, default=0.5)
    parser.add_argument('--wbf_weight', type=float, default=0.0)
    parser.add_argument('--cos_lr', action='store_true')    
    parser.add_argument('--CosineAnnealingLR', action='store_true')
    parser.add_argument('--CosineAnnealingWarmRestarts', action='store_true')    
    parser.add_argument('--teacher_earlystopping', action='store_true')    
    parser.add_argument('--patience', type=int, default=10, help='EarlyStopping patience (epochs without improvement)')
    
    parser.add_argument('--consistency_loss', action='store_true', help='Whether use the consistency loss (newly added)')
    parser.add_argument('--use_tuc', action='store_true')
    parser.add_argument('--use_eman', action='store_true')
    parser.add_argument('--alpha_weight', type=float, default=0.5, help='The weight for the consistency loss (newly added)')
    parser.add_argument('--gradient_thr', type=float, default=0.02)
    parser.add_argument('--moving_avg_diff_thr', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=5)

    parser.add_argument('--distill_loss', action='store_true')
    parser.add_argument('--contrastive',action='store_true')
    parser.add_argument('--cont_weight', type=float, default=0.05, help='The weight for the contrastive loss (newly added)')
    parser.add_argument('--domain_weight', type=float, default=0.1)

    parser.add_argument('--student_weight', type=str, default='None', help='Resuming weights path of student model in UMT')
    parser.add_argument('--teacher_weight', type=str, default='None', help='Resuming weights path of teacher model in UMT')
    parser.add_argument('--save_dir', type=str, default='None', help='Resuming project path in UMT')


    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run


    log_file_name = os.path.join(Path(opt.save_dir), f'{os.path.basename(Path(opt.save_dir))}_log.txt')

    if not os.path.exists(Path(opt.save_dir)):
        os.makedirs(Path(opt.save_dir))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                #'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0),    # segment copy-paste (probability)
                'loss_ota': (1, 0.0, 1.0),
                'loss_ota': (1, 0.0, 1.0),
                'sce_alpha': (0.5, 1.0, 2.0),
                'sce_beta': (1, 0.5, 1.0),
                'asl_pos':(1, 2, 3),
                'asl_neg':(4, 3, 1)
                }
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                #w = fitness(x) - fitness(x).min()  # weights
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
