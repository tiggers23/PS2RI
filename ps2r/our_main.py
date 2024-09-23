#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import argparse
import csv
import logging
import os
import json
import random
import pickle
import sys
from global_config import *
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    AlbertForSequenceClassification,
    BertForNextSentencePrediction,
    BertTokenizer,
    get_cosine_schedule_with_warmup,
)
# from new_models1 import *
from ps2r_models import *
from sar_models import Sarcasm
from transformers.optimization import AdamW
import time
from copy import deepcopy
import pandas as pd
from openpyxl import load_workbook
import warnings
warnings.filterwarnings("ignore")

def return_unk():
    return 0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, choices=["language_only", "acoustic_only", "visual_only", "our"], default="our",
)

parser.add_argument("--dataset", type=str, choices=["sarcasm"], default="sarcasm")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=85)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_heads", type=int, default=12)
parser.add_argument("--modal_n_layers", type=int, default=4)
parser.add_argument("--sar_n_layers", type=int, default=4)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--sar_cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=192)
parser.add_argument("--sar_fusion_dim", type=int, default=192)
parser.add_argument("--fc_dim", type=int, default=768)
parser.add_argument("--dropout", type=float, default=0.4)
parser.add_argument("--sar_dropout", type=float, default=0.4)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--text_dim", type=int, default=768)
parser.add_argument("--file_path", type=str, default='out4')

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--learning_rate", type=float, default=0.000005)
parser.add_argument("--learning_rate_a", type=float, default=0.003)
parser.add_argument("--learning_rate_h", type=float, default=0.0000005)
parser.add_argument("--learning_rate_v", type=float, default=0.003)
parser.add_argument("--warmup_ratio", type=float, default=0.07178)
parser.add_argument("--save_weight", type=str, choices=["True", "False"], default="False")
parser.add_argument("--weight", type=str)
parser.add_argument("--save_path", type=str)
parser.add_argument("--device", type=str)

args = parser.parse_args()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, visual, acoustic, sar, si, se, ei, ee):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.visual = visual
        self.acoustic = acoustic
        self.sar = sar
        self.si = si
        self.se = se
        self.ei = ei
        self.ee = ee




def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    pop_count = 0
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            pop_count += 1
            tokens_a.pop(0)
    return pop_count


# albert tokenizer split words in to subwords. "_" marker helps to find thos sub words
# our acoustic and visual features are aligned on word level. So we just create copy the same
# visual/acoustic vectors that belong to same word.
def get_inversion(tokens, SPIECE_MARKER="▁"):
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    return inversions


def convert_sarcasm_to_features(examples, tokenizer, sarcasm=False, sentiment=False,punchline_only=False):
    features = []

    for (ex_index, example) in enumerate(examples):

        # p denotes punchline, c deontes context
        (
            ((p_words, p_visual, p_acoustic, p_hcf),
            (c_words, c_visual, c_acoustic, c_hcf),
            hid,
            sar),
            si,
            se,
            ei,
            ee
        ) = example
        if (sarcasm and sar==1) or (sentiment and sar==0) or (sarcasm == False and sar==0) or (sentiment == False and sar == 1):
            text_a = ". ".join(c_words)
            text_b = p_words + "."
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)

            inversions_a = get_inversion(tokens_a)
            inversions_b = get_inversion(tokens_b)

            pop_count = _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

            inversions_a = inversions_a[pop_count:]
            inversions_b = inversions_b[: len(tokens_b)]

            visual_a = []
            acoustic_a = []
            for inv_id in inversions_a:
                visual_a.append(c_visual[inv_id, :])
                acoustic_a.append(c_acoustic[inv_id, :])

            visual_a = np.array(visual_a)
            acoustic_a = np.array(acoustic_a)

            visual_b = []
            acoustic_b = []
            for inv_id in inversions_b:
                visual_b.append(p_visual[inv_id, :])
                acoustic_b.append(p_acoustic[inv_id, :])

            visual_b = np.array(visual_b)
            acoustic_b = np.array(acoustic_b)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

            acoustic_zero = np.zeros((1, ACOUSTIC_DIM_ALL))
            if len(tokens_a) == 0:
                acoustic = np.concatenate(
                    (acoustic_zero, acoustic_zero, acoustic_b, acoustic_zero)
                )
            else:
                acoustic = np.concatenate(
                    (acoustic_zero, acoustic_a, acoustic_zero, acoustic_b, acoustic_zero)
                )

            visual_zero = np.zeros((1, VISUAL_DIM_ALL))
            if len(tokens_a) == 0:
                visual = np.concatenate((visual_zero, visual_zero, visual_b, visual_zero))
            else:
                visual = np.concatenate(
                    (visual_zero, visual_a, visual_zero, visual_b, visual_zero)
                )

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            input_mask = [1] * len(input_ids)

            acoustic_padding = np.zeros(
                (args.max_seq_length - len(input_ids), acoustic.shape[1])
            )
            acoustic = np.concatenate((acoustic, acoustic_padding))
            acoustic = np.take(acoustic, acoustic_features_list, axis=1)

            visual_padding = np.zeros(
                (args.max_seq_length - len(input_ids), visual.shape[1])
            )
            visual = np.concatenate((visual, visual_padding))
            visual = np.take(visual, visual_features_list, axis=1)

            padding = [0] * (args.max_seq_length - len(input_ids))

            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == args.max_seq_length
            assert len(input_mask) == args.max_seq_length
            assert len(segment_ids) == args.max_seq_length
            assert acoustic.shape[0] == args.max_seq_length
            assert visual.shape[0] == args.max_seq_length


            si = si+1
            se = se+1
            ei = ei-1
            ee = ee-1

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    visual=visual,
                    acoustic=acoustic,
                    sar=sar,
                    si=si,
                    se=se,
                    ei=ei,
                    ee=ee
                )
            )

    return features


def get_appropriate_dataset(data, tokenizer, parition, sarcasm=False, sentiment=False):
    features = convert_sarcasm_to_features(data, tokenizer, sarcasm, sentiment)
    print(sarcasm, '  ', sentiment, '    ', len(features))
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    all_sar_label = torch.tensor([f.sar for f in features], dtype=torch.int)
    all_si_label = torch.tensor([f.si for f in features], dtype=torch.int)
    all_se_label = torch.tensor([f.se for f in features], dtype=torch.int)
    all_ei1_label = []
    all_ei2_label = []
    all_ei3_label = []
    all_ei4_label = []
    all_ei5_label = []
    all_ei6_label = []
    all_ei7_label = []
    all_ei8_label = []
    all_ei9_label = []

    all_ee1_label = []
    all_ee2_label = []
    all_ee3_label = []
    all_ee4_label = []
    all_ee5_label = []
    all_ee6_label = []
    all_ee7_label = []
    all_ee8_label = []
    all_ee9_label = []
    for f in features:
        for i in range(9):
            if i == f.ei:
                eval(f'all_ei{i + 1}_label.append(1)')
            else:
                eval(f'all_ei{i + 1}_label.append(0)')

            if i == f.ee:
                eval(f'all_ee{i + 1}_label.append(1)')
            else:
                eval(f'all_ee{i + 1}_label.append(0)')
    all_ei1_label = torch.tensor(all_ei1_label,dtype=torch.int)
    all_ei2_label = torch.tensor(all_ei2_label,dtype=torch.int)
    all_ei3_label = torch.tensor(all_ei3_label,dtype=torch.int)
    all_ei4_label = torch.tensor(all_ei4_label,dtype=torch.int)
    all_ei5_label = torch.tensor(all_ei5_label,dtype=torch.int)
    all_ei6_label = torch.tensor(all_ei6_label,dtype=torch.int)
    all_ei7_label = torch.tensor(all_ei7_label,dtype=torch.int)
    all_ei8_label = torch.tensor(all_ei8_label,dtype=torch.int)
    all_ei9_label = torch.tensor(all_ei9_label,dtype=torch.int)

    all_ee1_label = torch.tensor(all_ee1_label,dtype=torch.int)
    all_ee2_label = torch.tensor(all_ee2_label,dtype=torch.int)
    all_ee3_label = torch.tensor(all_ee3_label,dtype=torch.int)
    all_ee4_label = torch.tensor(all_ee4_label,dtype=torch.int)
    all_ee5_label = torch.tensor(all_ee5_label,dtype=torch.int)
    all_ee6_label = torch.tensor(all_ee6_label,dtype=torch.int)
    all_ee7_label = torch.tensor(all_ee7_label,dtype=torch.int)
    all_ee8_label = torch.tensor(all_ee8_label,dtype=torch.int)
    all_ee9_label = torch.tensor(all_ee9_label,dtype=torch.int)


    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_sar_label,
        all_si_label,
        all_se_label,
        all_ei1_label,
        all_ei2_label,
        all_ei3_label,
        all_ei4_label,
        all_ei5_label,
        all_ei6_label,
        all_ei7_label,
        all_ei8_label,
        all_ei9_label,
        all_ee1_label,
        all_ee2_label,
        all_ee3_label,
        all_ee4_label,
        all_ee5_label,
        all_ee6_label,
        all_ee7_label,
        all_ee8_label,
        all_ee9_label
    )

    return dataset


def set_up_data_loader():
    if args.dataset == "humor":
        data_file = "ur_funny.pkl"
    elif args.dataset == "sarcasm":
        data_file = "emotion_data.pickle"

    with open(
            os.path.join(os.path.join(DATASET_LOCATION, args.dataset),data_file),
            "rb",
    ) as handle:
        all_data = pickle.load(handle)

    train_data = all_data["train"]
    dev_data = all_data["dev"]
    dev_data_sarcasm = deepcopy(all_data["dev"])
    dev_data_sentiment = deepcopy(all_data["dev"])
    test_data = all_data["test"]
    test_data_sarcasm = deepcopy(all_data["test"])
    test_data_sentiment = deepcopy(all_data["test"])

    tokenizer = AlbertTokenizer.from_pretrained("./albert-base-v2")

    train_dataset = get_appropriate_dataset(train_data, tokenizer, "train")
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer, "dev")
    dev_dataset_sarcasm = get_appropriate_dataset(dev_data_sarcasm, tokenizer, "dev", sarcasm=True)
    dev_dataset_sentiment = get_appropriate_dataset(dev_data_sentiment, tokenizer, "dev", sentiment=True)
    test_dataset = get_appropriate_dataset(test_data, tokenizer, "test")
    test_dataset_sarcasm = get_appropriate_dataset(test_data_sarcasm, tokenizer, "test", sarcasm=True)
    test_dataset_sentiment = get_appropriate_dataset(test_data_sentiment, tokenizer, "test", sentiment=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    dev_dataloader_sarcasm = DataLoader(
        dev_dataset_sarcasm, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    dev_dataloader_sentiment = DataLoader(
        dev_dataset_sentiment, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    test_dataloader_sarcasm = DataLoader(
        test_dataset_sarcasm, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    test_dataloader_sentiment = DataLoader(
        test_dataset_sentiment, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    return train_dataloader, dev_dataloader, dev_dataloader_sarcasm, dev_dataloader_sentiment, test_dataloader, test_dataloader_sarcasm, test_dataloader_sentiment


def train_epoch(model, sar_model, train_dataloader, optimizer, scheduler):
    model.train()
    sar_model.eval()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    outputs=None
    for step, batch in tqdm(enumerate(train_dataloader)):

        batch = tuple(t.to(args.device) for t in batch)
        (
            input_ids,
            visual,
            acoustic,
            input_mask,
            segment_ids,
            sar_ids,
            si_ids,
            se_ids,
            all_ei1_ids,
            all_ei2_ids,
            all_ei3_ids,
            all_ei4_ids,
            all_ei5_ids,
            all_ei6_ids,
            all_ei7_ids,
            all_ei8_ids,
            all_ei9_ids,
            all_ee1_ids,
            all_ee2_ids,
            all_ee3_ids,
            all_ee4_ids,
            all_ee5_ids,
            all_ee6_ids,
            all_ee7_ids,
            all_ee8_ids,
            all_ee9_ids
        ) = batch

        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        if args.model == "language_only":
            outputs = model(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
        elif args.model == "acoustic_only":
            outputs = model(
                acoustic
            )
        elif args.model == "visual_only":
            outputs = model(
                visual
            )
        elif args.model == "our":
            model = model.cpu()
            sar_model.to(args.device)
            sarcasm_out,_ = sar_model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask)
            sar_model = sar_model.cpu()
            model.to(args.device)
            outputs = model(input_ids, visual, acoustic, sarcasm_out, token_type_ids=segment_ids, attention_mask=input_mask, )

        si_out = outputs



        loss = F.cross_entropy(si_out, si_ids.long())

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        loss.backward()

        for o_i in range(len(optimizer)):
            optimizer[o_i].step()
            scheduler[o_i].step()

        model.zero_grad()

    return tr_loss / nb_tr_steps



def eval_epoch(model, sar_model, data_loader):
    """ Epoch operation in evaluation phase """
    model.eval()
    sar_model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0

    si_preds = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader)):
        # for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):

            batch = tuple(t.to(args.device) for t in batch)
            (
                input_ids,
                visual,
                acoustic,
                input_mask,
                segment_ids,
                sar_ids,
                si_ids,
                se_ids,
                ei1_ids,
                ei2_ids,
                ei3_ids,
                ei4_ids,
                ei5_ids,
                ei6_ids,
                ei7_ids,
                ei8_ids,
                ei9_ids,
                ee1_ids,
                ee2_ids,
                ee3_ids,
                ee4_ids,
                ee5_ids,
                ee6_ids,
                ee7_ids,
                ee8_ids,
                ee9_ids
            ) = batch

            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            if args.model == "language_only":
                outputs = model(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=None,
                )
            elif args.model == "acoustic_only":
                outputs = model(
                    acoustic
                )
            elif args.model == "visual_only":
                outputs = model(
                    visual
                )
            elif args.model == "our":
                # model = model.cpu()
                sar_model.to(args.device)
                model.to(args.device)
                sarcasm_out,_ = sar_model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                                                 attention_mask=input_mask)
                outputs = model(input_ids, visual, acoustic, sarcasm_out, token_type_ids=segment_ids,
                                attention_mask=input_mask, )
            si_out = outputs

            tmp_eval_loss = F.cross_entropy(si_out, si_ids.long())

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if len(si_preds) == 0:
                si_preds = si_out.detach().cpu()
                si_labels = si_ids.detach().cpu()
            else:
                si_preds = torch.cat([ si_preds, si_out.detach().cpu() ], dim=0)
                si_labels = torch.cat([ si_labels, si_ids.detach().cpu() ], dim=0)

        eval_loss = eval_loss / nb_eval_steps

    return si_preds, si_labels, eval_loss


def test_score_model(model, sar_model, data_loader):
    se_preds, se_labels, test_loss = eval_epoch(model, sar_model, data_loader)

    se_acc = accuracy_score(se_labels.cpu(), torch.argmax(se_preds, -1))
    se_f1 = f1_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1), labels=[0, 1, 2],average='micro')
    se_precision = precision_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1), labels=[0, 1, 2],average='micro')
    se_recall = recall_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1), labels=[0, 1, 2],average='micro')
    se_f1_macro = f1_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1).cpu(), labels=[0, 1, 2], average='macro')
    se_precision_macro = precision_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1).cpu(), labels=[0, 1, 2],average='macro')
    se_recall_macro = recall_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1).cpu(), labels=[0, 1, 2],average='macro')
    se_f1_weight = f1_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1).cpu(), labels=[0, 1, 2], average='weighted')
    se_precision_weight = precision_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1).cpu(), labels=[0, 1, 2],average='weighted')
    se_recall_weight = recall_score(se_labels.cpu(), torch.argmax(se_preds.cpu(), -1).cpu(), labels=[0, 1, 2],average='weighted')
    print("\n--------------------sentiment explicit prediction--------------------")
    print(f"acc { format( se_acc , '.4f') } | "
          f"Pre { format( se_precision , '.4f') } | "
          f"R { format(se_recall, '.4f') } | "
          f"F1 { format(se_f1, '.4f') } \n"
          f"Weigfhted:  Pre { format( se_precision_weight , '.4f') } | "
          f"R { format(se_recall_weight, '.4f') } | "
          f"F1 { format(se_f1_weight, '.4f') } |")

    return se_acc, se_precision, se_recall, se_f1, se_precision_macro, se_recall_macro, se_f1_macro, se_precision_weight, se_recall_weight, se_f1_weight, test_loss


def train(
        model,
        sar_model,
        train_dataloader,
        dev_dataloader,
        dev_dataloader_sarcasm,
        dev_dataloader_sentiment,
        test_dataloader,
        test_dataloader_sarcasm,
        test_dataloader_sentiment,
        optimizer,
        scheduler,
        args
):
    best_valid_f1 = 0
    valid_losses = []
    best_epoch = 0
    best_valid_loss = 1e9
    best_test_acc, best_test_pre, best_test_recall, best_test_f1, best_test_pre_m, best_test_recall_m, best_test_f1m = (None, None, None, None, None, None, None)
    n_epochs = args.epochs
    start_time = time.time()
    for epoch_i in range(n_epochs):
        train_time = time.time()
        train_loss = train_epoch(model, sar_model, train_dataloader, optimizer, scheduler)
        print(
            "\nepoch:{},train_loss: {}, train_time:{} s".format(
                epoch_i, format(train_loss, '.4f'), format(time.time()-train_time, '.2f')
            )
        )
        # print(f"------------------- Dev epoch {epoch_i} -------------------")
        dev_sar_acc, dev_precision, dev_recall, dev_f1, dev_precision_macro, dev_recall_macro, dev_f1_macro,dev_precision_weight, dev_recall_weight, dev_f1_weight, dev_loss = test_score_model(
            model, sar_model, dev_dataloader)
        dev_sarcasm_sar_acc, dev_sarcasm_precision, dev_sarcasm_recall, dev_sarcasm_f1, dev_sarcasm_precision_macro, dev_sarcasm_recall_macro, dev_sarcasm_f1_macro, _,_,_, dev_sarcasm_loss = test_score_model(
            model, sar_model, dev_dataloader_sarcasm)
        dev_sentiment_sar_acc, dev_sentiment_precision, dev_sentiment_recall, dev_sentiment_f1, dev_sentiment_precision_macro, dev_sentiment_recall_macro, dev_sentiment_f1_macro, _,_,_, dev_sentiment_loss = test_score_model(
            model, sar_model, dev_dataloader_sentiment)

        valid_losses.append(dev_loss)

        # print(f"------------------- Test epoch {epoch_i} -------------------")
        test_sar_acc, test_precision, test_recall, test_f1, test_precision_macro, test_recall_macro, test_f1_macro, test_precision_weight, test_recall_weight, test_f1_weight, test_loss = test_score_model(
            model, sar_model, test_dataloader)
        test_sarcasm_sar_acc, test_sarcasm_precision, test_sarcasm_recall, test_sarcasm_f1, test_sarcasm_precision_macro, test_sarcasm_recall_macro, test_sarcasm_f1_macro, test_sarcasm_precision_weight, test_sarcasm_recall_weight, test_sarcasm_f1_weight, test_sarcasm_loss = test_score_model(
            model, sar_model, test_dataloader_sarcasm)
        test_sentiment_sar_acc, test_sentiment_precision, test_sentiment_recall, test_sentiment_f1, test_sentiment_precision_macro, test_sentiment_recall_macro, test_sentiment_f1_macro, test_sentiment_precision_weight, test_sentiment_recall_weight, test_sentiment_f1_weight, test_sentiment_loss = test_score_model(
            model, sar_model, test_dataloader_sentiment)

        if (dev_f1_weight >= best_valid_f1):
            best_valid_loss = dev_loss
            best_valid_f1 = dev_f1_weight
            best_epoch = epoch_i
            best_test_acc = test_sar_acc
            best_test_pre = test_precision
            best_test_recall = test_recall
            best_test_f1 = test_f1
            best_test_pre_m = test_precision_macro
            best_test_recall_m = test_recall_macro
            best_test_f1m = test_f1_macro
            best_test_pre_w = test_precision_weight
            best_test_recall_w = test_recall_weight
            best_test_f1_w = test_f1_weight


            best_test_sarcasm_acc = test_sarcasm_sar_acc
            best_test_sarcasm_pre = test_sarcasm_precision
            best_test_sarcasm_recall = test_sarcasm_recall
            best_test_sarcasm_f1 = test_sarcasm_f1
            best_test_sarcasm_pre_m = test_sarcasm_precision_macro
            best_test_sarcasm_recall_m = test_sarcasm_recall_macro
            best_test_sarcasm_f1m = test_sarcasm_f1_macro
            best_test_sarcasm_pre_w = test_sarcasm_precision_weight
            best_test_sarcasm_recall_w = test_sarcasm_recall_weight
            best_test_sarcasm_f1_w = test_sarcasm_f1_weight

            best_test_sentiment_acc = test_sentiment_sar_acc
            best_test_sentiment_pre = test_sentiment_precision
            best_test_sentiment_recall = test_sentiment_recall
            best_test_sentiment_f1 = test_sentiment_f1
            best_test_sentiment_pre_m = test_sentiment_precision_macro
            best_test_sentiment_recall_m = test_sentiment_recall_macro
            best_test_sentiment_f1m = test_sentiment_f1_macro
            best_test_sentiment_pre_w = test_sentiment_precision_weight
            best_test_sentiment_recall_w = test_sentiment_recall_weight
            best_test_sentiment_f1_w = test_sentiment_f1_weight

            if (args.save_weight == "True"):
                if not os.path.exists(f'./output/{args.save_path}'):
                    os.mkdir(f'./output/{args.save_path}')
                torch.save(model.state_dict(), f'./output/{args.save_path}' + run_name + '.pt')


        log_stats = {
                "epoch":epoch_i,
                "train_loss": train_loss,
                "valid_loss": dev_loss,
                "test_loss": test_loss,
                "best_valid_loss": best_valid_loss,
                "dev_accuracy": dev_sar_acc,
                "dev_precision": dev_precision,
                "dev_recall": dev_recall,
                "dev_f1": dev_f1,
                "dev_precision_m": dev_precision_macro,
                "dev_recall_m": dev_recall_macro,
                "dev_f1_m": dev_f1_macro,
                "dev_sarcasm_accuracy": dev_sarcasm_sar_acc,
                "dev_sarcasm_precision": dev_sarcasm_precision,
                "dev_sarcasm_recall": dev_sarcasm_recall,
                "dev_sarcasm_f1": dev_sarcasm_f1,
                "dev_sarcasm_precision_m": dev_sarcasm_precision_macro,
                "dev_sarcasm_recall_m": dev_sarcasm_recall_macro,
                "dev_sarcasm_f1_m": dev_sarcasm_f1_macro,
                "dev_sentiment_accuracy": dev_sentiment_sar_acc,
                "dev_sentiment_precision": dev_sentiment_precision,
                "dev_sentiment_recall": dev_sentiment_recall,
                "dev_sentiment_f1": dev_sentiment_f1,
                "dev_sentiment_precision_m": dev_sentiment_precision_macro,
                "dev_sentiment_recall_m": dev_sentiment_recall_macro,
                "dev_sentiment_f1_m": dev_sentiment_f1_macro,
                "test_accuracy": test_sar_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_precision_m": test_precision_macro,
                "test_recall_m": test_recall_macro,
                "test_f1_m": test_f1_macro,
                "test_sentiment_accuracy": test_sentiment_sar_acc,
                "test_sentiment_precision": test_sentiment_precision,
                "test_sentiment_recall": test_sentiment_recall,
                "test_sentiment_f1": test_sentiment_f1,
                "test_sentiment_precision_m": test_sentiment_precision_macro,
                "test_sentiment_recall_m": test_sentiment_recall_macro,
                "test_sentiment_f1_m": test_sentiment_f1_macro,
                "test_sarcasm_accuracy": test_sarcasm_sar_acc,
                "test_sarcasm_precision": test_sarcasm_precision,
                "test_sarcasm_recall": test_sarcasm_recall,
                "test_sarcasm_f1": test_sarcasm_f1,
                "test_sarcasm_precision_m": test_sarcasm_precision_macro,
                "test_sarcasm_recall_m": test_sarcasm_recall_macro,
                "test_sarcasm_f1_m": test_sarcasm_f1_macro,
            }
        with open(f'./output/si/{args.file_path}/{args.save_path}.txt', "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    print(f"\nTotal time is {format((time.time()-start_time)/60,'.2f')} min")


def get_optimizer_scheduler(params, num_training_steps, learning_rate=1e-5):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def prep_for_training(num_training_steps):
    if args.model == "language_only":
        model = AlbertForSequenceClassification.from_pretrained(
            "./albert-base-v2", num_labels=1
        )
    elif args.model == "acoustic_only":
        model = Transformer(ACOUSTIC_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)

    elif args.model == "visual_only":
        model = Transformer(VISUAL_DIM, num_layers=args.n_layers, nhead=args.n_heads, dim_feedforward=args.fc_dim)
    if args.model == "our":
        visual_model = Transformer(VISUAL_DIM, num_layers=8, nhead=4, dim_feedforward=1024)
        sar_visual_model = Transformer(VISUAL_DIM, num_layers=8, nhead=4, dim_feedforward=1024)
        visual_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmVisualTransformer.pt"))
        acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=1, nhead=3, dim_feedforward=512)
        sar_acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=1, nhead=3, dim_feedforward=512)
        acoustic_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmAcousticTransformer.pt"))
        text_model = AlbertModel.from_pretrained('./albert-base-v2')
        sar_text_model = AlbertModel.from_pretrained('./albert-base-v2')
        sarcasm_model = Sarcasm(sar_text_model, sar_visual_model, sar_acoustic_model, args, fusion_dim=args.sar_fusion_dim)
        sarcasm_model.load_state_dict(torch.load(f'./output/sar_weight/{args.weight}' + '.pt'))
        model = Ours(text_model, visual_model, acoustic_model, args, fusion_dim=args.fusion_dim)
    else:
        raise ValueError("Requested model is not available")

    model.to(args.device)

    loss_fct = BCEWithLogitsLoss()

    # Prepare optimizer
    # used different learning rates for different componenets.

    if args.model == 'our':
        acoustic_params, visual_params, other_params = model.get_params()
        optimizer_o, scheduler_o = get_optimizer_scheduler(other_params, num_training_steps,
                                                           learning_rate=args.learning_rate)
        optimizer_v, scheduler_v = get_optimizer_scheduler(visual_params, num_training_steps,
                                                           learning_rate=args.learning_rate_v)
        optimizer_a, scheduler_a = get_optimizer_scheduler(acoustic_params, num_training_steps,
                                                           learning_rate=args.learning_rate_a)

        optimizers = [optimizer_o, optimizer_v, optimizer_a]
        schedulers = [scheduler_o, scheduler_v, scheduler_a]
    else:
        params = list(model.named_parameters())

        optimizer_l, scheduler_l = get_optimizer_scheduler(
            params, num_training_steps, learning_rate=args.learning_rate
        )

        optimizers = [optimizer_l]
        schedulers = [scheduler_l]


    return model, sarcasm_model, optimizers, schedulers, loss_fct


def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def main():

    if (args.seed == -1):
        seed = random.randint(0, 9999)
        print("seed", seed)
    else:
        seed = args.seed

    set_random_seed(seed)

    train_dataloader, dev_dataloader, dev_dataloader_sarcasm, dev_dataloader_sentiment, test_dataloader, test_dataloader_sarcasm, test_dataloader_sentiment = set_up_data_loader()
    print("Dataset Loaded: ", args.dataset)
    num_training_steps = len(train_dataloader) * args.epochs

    model, sar_model, optimizers, schedulers, loss_fct = prep_for_training(
        num_training_steps
    )
    print("Model Loaded: ", args.model)
    train(
        model,
        sar_model,
        train_dataloader,
        dev_dataloader,
        dev_dataloader_sarcasm,
        dev_dataloader_sentiment,
        test_dataloader,
        test_dataloader_sarcasm,
        test_dataloader_sentiment,
        optimizers,
        schedulers,
        args
    )



if __name__ == "__main__":
    if not os.path.exists(f'./output/si/out3/{args.save_path}.txt'):
        main()#!/usr/bin/env python2