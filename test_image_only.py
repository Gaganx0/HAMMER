# test_image_only.py
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from ruamel import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.HAMMER import HAMMER

from types import MethodType
import logging


# ---------------- utils ----------------
def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers on re-runs
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set, idx=idx, loss=loss, acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)
    return logger


def is_text_only_label(s: str) -> bool:
    if s is None:
        return False
    s = s.lower()
    # Common variants seen in your code/papers
    return s in {"ts", "ta", "text_swap", "text- swap", "text attribute",
                 "text_attribute", "text-attribute", "tattr", "tswap"}


def is_image_manip_label(s: str) -> bool:
    if s is None:
        return False
    s = s.lower()
    # Image-side manipulations typical in HAMMER
    return s in {"fs", "fa", "face_swap", "face- swap", "face_attribute",
                 "face_attribute_change", "faceswap", "face-attribute"}


def build_empty_text_batch(tokenizer, batch_size: int, device):
    """
    Make a minimal BatchEncoding that keeps shapes valid but conveys 'no text'.
    We purposely pass empty strings so word_ids() exists, then we zero masks.
    """
    dummy = [""] * batch_size
    enc = tokenizer(
        dummy,
        max_length=128,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    # Zero out attention so the text encoder (if called) effectively no-ops.
    enc["attention_mask"] = torch.zeros_like(torch.tensor(enc["attention_mask"]))
    # Move to device as tensors
    enc.input_ids = torch.LongTensor(enc["input_ids"]).to(device)
    enc.attention_mask = enc["attention_mask"].to(device)
    return enc


def text_input_adjust(text_input, fake_word_pos, device):
    """
    Original adapter kept intact. Safe for empty strings (word_ids -> [None, None]).
    """
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids]) - 1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    subword_idx_rm_CLSSEP_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() if hasattr(fake_word_pos[i], "numpy") else []

        subword_idx = text_input.word_ids(i)  # may be [None, None] for empty text
        subword_idx_rm_CLSSEP = subword_idx[1:-1] if subword_idx is not None else []
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)
        subword_idx_rm_CLSSEP_batch.append(subword_idx_rm_CLSSEP_array)

        for j in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == j)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch, subword_idx_rm_CLSSEP_batch


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    model.eval()
    header = 'Evaluation:'

    print('>>> Computing features for evaluation (IMAGE-ONLY = %s)...' % str(args.image_only))
    print_freq = 200

    y_true, y_pred = [], []

    # IOU
    IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], []

    # token metrics (will be skipped in image-only)
    TP_all = TN_all = FP_all = FN_all = 0

    # multi-label
    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    cls_nums_all = 0
    cls_acc_all = 0

    # For multiclass per-head F1 (FS, FA, TS, TA ordering expected downstream)
    TP_all_multicls = np.zeros(4, dtype=int)
    TN_all_multicls = np.zeros(4, dtype=int)
    FP_all_multicls = np.zeros(4, dtype=int)
    FN_all_multicls = np.zeros(4, dtype=int)

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, ids, image_dir_all) in \
            enumerate(utils.MetricLogger(delimiter="  ").log_every(args, data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)

        # ----- TEXT HANDLING -----
        if args.image_only:
            # empty text batch to keep shapes; attention all zeros → no text signal
            text_input = build_empty_text_batch(tokenizer, batch_size=len(image), device=device)
            fake_token_pos = [[] for _ in range(len(image))]
        else:
            if isinstance(text, tuple):
                text = list(text)
            text_input = tokenizer(
                text, max_length=128, truncation=True,
                add_special_tokens=True, return_attention_mask=True,
                return_token_type_ids=False
            )
            text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        # ----- MODEL FORWARD -----
        # We pass everything in, but in image_only mode the text has zero attention.
        logits_real_fake, logits_multicls, output_coord, logits_tok = model(
            image, label, text_input, fake_image_box, fake_token_pos, is_train=False
        )

        # ----- BINARY RF METRICS (IMAGE-ONLY: drop TS/TA) -----
        cls_label = torch.ones(len(label), dtype=torch.long, device=image.device)
        real_pos = [idx for idx, l in enumerate(label) if str(l).lower() in {"orig", "real", "genuine", "natural"}]
        cls_label[real_pos] = 0

        if args.image_only:
            valid_indices = [idx for idx, l in enumerate(label) if not is_text_only_label(str(l))]
        else:
            valid_indices = list(range(len(label)))

        if len(valid_indices) > 0:
            logits_rf_valid = F.softmax(logits_real_fake[valid_indices], dim=1)[:, 1]
            y_pred.extend(logits_rf_valid.detach().cpu().tolist())
            y_true.extend(cls_label[valid_indices].detach().cpu().tolist())

            pred_acc = logits_real_fake[valid_indices].argmax(1)
            cls_nums_all += len(valid_indices)
            cls_acc_all += torch.sum(pred_acc == cls_label[valid_indices]).item()

        # ----- MULTI-LABEL METRICS -----
        # Keep AP available for FS/FA; TS/TA still flow through but we’ll call out that text classes are meaningless in image-only.
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)

        # crude per-class F1 accumulation (expects order [FS, FA, TS, TA])
        for cls_idx in range(logits_multicls.shape[1]):
            cls_pred = (logits_multicls[:, cls_idx] >= 0).long()
            TP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 1)).item()
            TN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 0)).item()
            FP_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 0) * (cls_pred == 1)).item()
            FN_all_multicls[cls_idx] += torch.sum((target[:, cls_idx] == 1) * (cls_pred == 0)).item()

        # ----- BBOX IOU (IMAGE-ONLY: evaluate on FS/FA only) -----
        with torch.no_grad():
            boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
            boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

            if args.image_only:
                img_idx = [idx for idx, l in enumerate(label) if is_image_manip_label(str(l))]
            else:
                img_idx = list(range(len(label)))

            if len(img_idx) > 0:
                IOU, _ = box_ops.box_iou(boxes1[img_idx], boxes2.to(device)[img_idx], test=True)
                IOU_pred.extend(IOU.detach().cpu().tolist())

                IOU_50_bt = (IOU > 0.5).long()
                IOU_75_bt = (IOU > 0.75).long()
                IOU_95_bt = (IOU > 0.95).long()

                IOU_50.extend(IOU_50_bt.cpu().tolist())
                IOU_75.extend(IOU_75_bt.cpu().tolist())
                IOU_95.extend(IOU_95_bt.cpu().tolist())

        # ----- TOKEN METRICS (SKIPPED IN IMAGE-ONLY) -----
        if not args.image_only:
            token_label = text_input.attention_mask[:, 1:].clone()  # ignore CLS
            token_label[token_label == 0] = -100
            token_label[token_label == 1] = 0
            for b in range(len(fake_token_pos)):
                for pos in fake_token_pos[b]:
                    if pos < token_label.shape[1]:
                        token_label[b, pos] = 1

            logits_tok_reshape = logits_tok.view(-1, 2)
            logits_tok_pred = logits_tok_reshape.argmax(1)
            token_label_reshape = token_label.view(-1)

            TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
            TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
            FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
            FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    # ===== Aggregate =====
    # RF
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if cls_nums_all > 0:
        ACC_cls = cls_acc_all / cls_nums_all
    else:
        ACC_cls = 0.0

    if len(set(y_true.tolist())) > 1:
        from sklearn.metrics import roc_auc_score, roc_curve
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        AUC_cls = float(roc_auc_score(y_true, y_pred))
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        EER_cls = float(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))
    else:
        AUC_cls, EER_cls = 0.0, 0.0

    # BBOX
    if len(IOU_pred) > 0:
        IOU_score = sum(IOU_pred) / len(IOU_pred)
        IOU_ACC_50 = sum(IOU_50) / len(IOU_50)
        IOU_ACC_75 = sum(IOU_75) / len(IOU_75)
        IOU_ACC_95 = sum(IOU_95) / len(IOU_95)
    else:
        IOU_score = IOU_ACC_50 = IOU_ACC_75 = IOU_ACC_95 = 0.0

    # Token metrics
    if not args.image_only and (TP_all + TN_all + FP_all + FN_all) > 0:
        ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
        Precision_tok = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0
        Recall_tok = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0
        F1_tok = (2 * Precision_tok * Recall_tok) / (Precision_tok + Recall_tok) if (Precision_tok + Recall_tok) > 0 else 0
    else:
        ACC_tok = Precision_tok = Recall_tok = F1_tok = 0.0  # reported as 0.0 in image-only

    # Multi-label
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()

    F1_multicls = np.zeros(4)
    for cls_idx in range(4):
        P = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx]) if (TP_all_multicls[cls_idx] + FP_all_multicls[cls_idx]) > 0 else 0.0
        R = TP_all_multicls[cls_idx] / (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx]) if (TP_all_multicls[cls_idx] + FN_all_multicls[cls_idx]) > 0 else 0.0
        F1_multicls[cls_idx] = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    return (AUC_cls, ACC_cls, EER_cls,
            float(MAP), float(OP), float(OR), float(OF1), float(CP), float(CR), float(CF1),
            F1_multicls,
            IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95,
            ACC_tok, Precision_tok, Recall_tok, F1_tok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=777, type=int)

    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')

    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--log', default=False, action='store_true', help='Enable logging')
    parser.add_argument('--model_save_epoch', type=int, default=5)
    parser.add_argument('--token_momentum', default=False, action='store_true')
    parser.add_argument('--test_epoch', default='best', type=str)

    # NEW: image-only ablation
    parser.add_argument('--image_only', default=False, action='store_true',
                        help='Run HAMMER in image-only evaluation: ignore TS/TA for RF/bbox; skip token metrics; pass empty text.')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # device & seeds
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # logging
    eval_type = os.path.basename(config['val_file'][0]).split('.')[0]
    if eval_type == 'test':
        eval_type = 'all'
    log_dir = os.path.join(args.output_dir, args.log_num or "run", 'evaluation')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'shell_{eval_type}.txt')
    logger = setlogger(log_file)

    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')
        if args.image_only:
            logger.info('>>> IMAGE-ONLY ABLATION ENABLED: TS/TA ignored for RF/bbox; token metrics disabled.')

    # model
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)
    if args.log:
        print("Creating MAMMER")
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    # Optional: expose a hint to the model (no-op if HAMMER.py doesn’t read it)
    if not hasattr(model, "image_only"):
        try:
            model.image_only = args.image_only
        except Exception:
            pass

    model = model.to(device)

    # checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    if args.log:
        print('load checkpoint from %s' % args.checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    if args.log:
        print(msg)

    # data
    if args.log:
        print("Creating dataset")
    _, val_dataset = create_dataset(config)
    if args.distributed:
        samplers = utils.create_sampler([val_dataset], [True], args.world_size, args.rank) + [None]
    else:
        samplers = [None]

    val_loader = create_loader(
        [val_dataset],
        samplers,
        batch_size=[config['batch_size_val']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]

    # eval
    if args.log:
        print("Start evaluation")
    (AUC_cls, ACC_cls, EER_cls,
     MAP, OP, OR, OF1, CP, CR, CF1, F1_multicls,
     IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95,
     ACC_tok, Precision_tok, Recall_tok, F1_tok) = evaluation(args, model, val_loader, tokenizer, device, config)

    # stats
    val_stats = {
        "AUC_cls": f"{AUC_cls*100:.4f}",
        "ACC_cls": f"{ACC_cls*100:.4f}",
        "EER_cls": f"{EER_cls*100:.4f}",
        "MAP": f"{MAP*100:.4f}",
        "OP": f"{OP*100:.4f}",
        "OR": f"{OR*100:.4f}",
        "OF1": f"{OF1*100:.4f}",
        "CP": f"{CP*100:.4f}",
        "CR": f"{CR*100:.4f}",
        "CF1": f"{CF1*100:.4f}",
        "F1_FS": f"{F1_multicls[0]*100:.4f}",
        "F1_FA": f"{F1_multicls[1]*100:.4f}",
        # TS/TA are text-only; leave numbers but remember they are meaningless in image-only
        "F1_TS": f"{F1_multicls[2]*100:.4f}",
        "F1_TA": f"{F1_multicls[3]*100:.4f}",
        "IOU_score": f"{IOU_score*100:.4f}",
        "IOU_ACC_50": f"{IOU_ACC_50*100:.4f}",
        "IOU_ACC_75": f"{IOU_ACC_75*100:.4f}",
        "IOU_ACC_95": f"{IOU_ACC_95*100:.4f}",
        # token metrics are zeros when image_only
        "ACC_tok": f"{ACC_tok*100:.4f}",
        "Precision_tok": f"{Precision_tok*100:.4f}",
        "Recall_tok": f"{Recall_tok*100:.4f}",
        "F1_tok": f"{F1_tok*100:.4f}",
    }

    if utils.is_main_process():
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': args.test_epoch}
        with open(os.path.join(log_dir, f"results_{eval_type}.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    if args.log:
        print("Done. Image-only:", args.image_only)


if __name__ == '__main__':
    main()
