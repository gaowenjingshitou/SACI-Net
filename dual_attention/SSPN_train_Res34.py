# -*- coding: utf-8 -*-
"""
# @file name  : cifar_train.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2021-04-22
# @brief      : 模型训练主代码
"""

import os
import pretrainedmodels
import sys
# from SSPN_own import *


from timm import create_model
BASE_DIR=r'/home/gwj/Intussption_classification'
BASE_DIR1=r'/home/gwj/Intussption_classification/models'
BASE_DIR2=r'/home/gwj/Intussption_classification/model'
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR1)
sys.path.append(BASE_DIR2)
from sspn_resnet34 import SSFPN
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tools.model_trainerO import ModelTrainer

from tools.common_tools_origin import *

from timm.scheduler import create_scheduler
from config import cfg_c,update_config, update_cfg_name
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import autocast, GradScaler
from tools.my_loss import *
import torch.nn.functional as F
import cv2
from io import BytesIO
import cv2
from models.resnet_cifar10 import resnet20
from models.resnet_cifar10 import *
from config.cifar_config import cfg
from datetime import datetime
from datasets.cifar_longtail的副本O import CifarDataset
from tools.progressively_balance import ProgressiveSampler
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from pytorchtool import EarlyStopping
from tensorboardX import SummaryWriter
from utils.scheduler import *
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# setup_seed(12345)  # 先固定随机种子
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='code for diagnose intussusception')
parser.add_argument(
        "--cfg_c",
        help="decide which cfg to use",
        required=False,
        default="../config/Intussusception_2022_bbn.yaml",
        type=str,
    )
parser.add_argument("--tensorboard_events", type=str, default='../results/events/',
                    help="path to tensorboard events")
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',choices=['cosine', 'tanh', 'step', 'multistep', 'poly'],
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', default=0.00001, help='learning rate')
parser.add_argument('--patience_epochs', default=10, help='patience_epochs')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument("--lr_policy", type=str, default='cosine',
                        choices=['poly', 'step', 'multi_step', 'exponential', 'cosine', 'lambda','onecycle'],
                        help="learning rate scheduler policy")
parser.add_argument('--bs', default=32, help='training batch size')
parser.add_argument('--epochs', default=100)
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
# parser.add_argument('--data_root_dir', default=r"G:\deep_learning_data\cifar10",
#                     help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"../data/",
                    help="path to your dataset")
parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'Nadam','adam','AdamW','adamw','adadelta','rmsprop','rmsproptf','fusedadamw'],
                    help="choose optimizer")

parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
# 显卡配置
parser.add_argument("--gpu_id", type=str, default='0',
                    help="GPU ID")
parser.add_argument("--multi_gpu", action='store_true', default=False)
parser.add_argument('--seed', type=int, default=19920608, metavar='S',
                    help='random seed (default: 42)')




def log_stats_train(train_results, epoch):
    tag_value = {'training_resnet_own_loss': train_results['train_loss'],
                 'training_resnet_own_accuracy': train_results['train_accuracy']}
    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, epoch)




def log_stats_val(val_results, epoch):

    tag_value = {'validation_resnet_own_loss': val_results['val_loss'],
                 'validation_resnet_own_accuracy': val_results['val_accuracy']}
    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, epoch)

        
        

def check_data_dir(path_data):
    if not os.path.exists(path_data):
        print("文件夹不存在，请检查数据是否存放到data_dir变量:{}".format(path_data))

if __name__ == "__main__":
    args = parser.parse_args()
    update_config(cfg_c, args)
    update_cfg_name(cfg_c)  # modify the cfg.NAME

    # 显卡环境
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device",device)

    cfg.lr_init = args.lr if args.lr else cfg.lr_init
    cfg.train_bs = args.bs if args.bs else cfg.train_bs
    cfg.epochs = args.epochs if args.epochs else cfg.epochs

    # 日志 & Tensorboard
    # train_logger = logger.get_logger(opts.logs + opts.dataset)

    writer = SummaryWriter(log_dir=args.tensorboard_events)

   # update_config(cfgs, args)
   # update_cfg_name(cfgs)  # modify
    # step0: setting path
    train_dir=args.data_root_dir
    valid_dir=args.data_root_dir
    #train_dir = os.path.join(args.data_root_dir, "cifar10_train")#'/Users/hello/Downloads/cifar-10/cifar10_train'
    #valid_dir = os.path.join(args.data_root_dir, "cifar10_test")#'/Users/hello/Downloads/cifar-10/cifar10_test'
    check_data_dir(train_dir)
    check_data_dir(valid_dir)
    num_classes = 3
    # num_class_list=['normal','sleeve_sign','concentric_circle_sign']
    num_class_list = [3601, 926, 5345]
    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfgs": cfg_c,
        "device": device
    }

    # 创建logger
    #res_dir = os.path.join(BASE_DIR, "..", "..", "results")#'/Users/hello/PycharmProjects/MTX/src/../../results'
   # res_dir = os.path.join(BASE_DIR, "results_densnet_2022_0328")
   # res_dir = os.path.join(BASE_DIR, "results_wide_resnet50_0607AM")
    res_dir = os.path.join(BASE_DIR, "SSPN_0701")
    logger, log_dir = make_logger(res_dir)

    # step1： 数据集
    # 构建MyDataset实例， 构建DataLoder
    #train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, isTrain=True)
    #valid_data = CifarDataset(root_dir=valid_dir, transform=cfg.transforms_valid, isTrain=False)

   # train_data = CifarDataset(root_dir=train_dir,transform=cfg.transforms_train,mode="train")
   # valid_data = CifarDataset(root_dir=valid_dir, transform=cfg.transforms_train, mode="val")

    train_data = CifarDataset(root_dir=train_dir, transform=cfg.transforms_train, mode="train",do_fmix=False, do_cutmix=False)
    valid_data = CifarDataset(root_dir=valid_dir, transform=cfg.transforms_train, mode="val",do_fmix=False, do_cutmix=False)


    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)
    if cfg.pb: #true
        sampler_generator = ProgressiveSampler(train_data, cfg.epochs)#<tools.progressively_balance.ProgressiveSampler object at 0x7fea26815f40>

    # step2: 模型
    #model = resnet20()

    #model = resnet110()
    # trained_model = ResNet152()
   # model = FPN101()

  #  model = ResNet152_FPN_attention()
    model = SSFPN("resnet34",pretrained=True)
    #  print(model)

    
    
    
    

    # model = nn.Sequential(trained_model,  # [b, 512, 1, 1]
    #
    #                       nn.Linear(1000, 3),
    #
    #                       ).cuda()
    model.to(device)
    #config = 'coatnet-0'
   # image_size = (3, 224, 224)
    early_stopping = EarlyStopping(10, verbose=True)
    #config = 'coatnet-0'

    #model = CoAtNet(image_size[1], image_size[2], image_size[0], config=config, num_classes=3)
     # to device， cpu or gpu
    
    # step3: 损失函数、优化器
    if cfg.label_smooth:
        loss_f = LabelSmoothLoss(cfg.label_smooth_eps)
    else:
        
        #loss_f = FocalLoss()
        #loss_f=CB_loss()
        loss_f=MWNLoss(para_dict)
        #loss_f = nn.CrossEntropyLoss()
        #loss_f = nn.nn.BCELoss()
       
        
    # 优化器
    if args.optimizer == 'sgd':
        #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9,
                                #weight_decay=opts.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_init)
    if args.optimizer == 'AdamW':
        optimizer =torch.optim.AdamW(model.parameters(), lr=cfg.lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.weight_decay, amsgrad=False,
                      maximize=False)
    if args.optimizer == 'Nadam':
        optimizer = Nadam(model.parameters(), lr=cfg.lr_init, weight_decay=cfg.weight_decay, eps=1e-8)
    
    

    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)
    #scheduler = option(args,optimizer, args.lr_policy, args.max_epoch)
    scheduler, num_epochs = create_scheduler(args, optimizer)
    # step4: 迭代训练
    # 记录训练所采用的模型、损失函数、优化器、配置参数cfg
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
        cfg, loss_f, scheduler, optimizer, model))

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    print("start training")
    for epoch in range(cfg.epochs):
        if cfg.pb:
            sampler, _ = sampler_generator(epoch) #sampler就是第几张图像的索引
            train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=False,
                                      num_workers=cfg.workers,
                                      sampler=sampler)
        loss_train, acc_train, mat_train, path_error_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, device, cfg, logger)

    

        loss_valid, acc_valid, mat_valid, path_error_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, epoch,device)
        
        
        
        train_results = {"train_loss": loss_train, 'train_accuracy': acc_train}
        log_stats_train(train_results, epoch)
        
        # print("train_results",train_results["train_loss"])
        # print("train_accuracy", train_results["train_accuracy"])
        
        
        
        val_results = {'val_loss': loss_valid, 'val_accuracy': acc_valid}
        
        log_stats_val(val_results, epoch)
        
        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
                    format(epoch + 1, cfg.epochs, acc_train, acc_valid, loss_train, loss_valid,
                           optimizer.param_groups[0]["lr"]))
        scheduler.step(epoch)
        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        
        
        # 保存混淆矩阵图
        show_confMat(mat_train, train_data.names, "train", log_dir, epoch=epoch, verbose=epoch == cfg.epochs - 1)
        show_confMat(mat_valid, valid_data.names, "valid", log_dir, epoch=epoch, verbose=epoch == cfg.epochs - 1)
        # 保存loss曲线， acc曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        # 模型保存
        if best_acc < acc_valid or epoch == cfg.epochs - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            pkl_name = "checkpoint_densnet_{}.pkl".format(epoch) if epoch == cfg.epochs - 1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            torch.cuda.empty_cache()

            # 保存错误图片的路径
            err_ims_name = "error_imgs_{}.pkl".format(epoch) if epoch == cfg.epochs-1 else "error_imgs_best.pkl"
            path_err_imgs = os.path.join(log_dir, err_ims_name)
            error_info = {}
            error_info["train"] = path_error_train
            error_info["valid"] = path_error_valid
            pickle.dump(error_info, open(path_err_imgs, 'wb'))
        early_stopping(loss_valid, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
    logger.info("{} done, best acc: {} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))
        