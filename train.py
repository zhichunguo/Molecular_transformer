'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import EncoderDecoderDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from sklearn.metrics import roc_auc_score
import numpy as np



def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    step = 0
    for batch in tqdm(training_data, mininterval=2, desc='- (Training)   ', leave=False):
        step += 1

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos, label = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred, pred_label = model(src_seq, src_pos, tgt_seq, tgt_pos)
        # print(enc_output.shape)
        # raise TypeError
        # print(pred_label.shape)
        # print(label.shape)
        # raise TypeError
        # print(pred_label)
        # print(label)
        # print(pred_label.shape)

        # pred_label = pred_label.reshape([pred_label.size[0]])
        pred_label = torch.squeeze(pred_label)
        
        label_loss = F.binary_cross_entropy_with_logits(pred_label, label)
        print(label_loss)
        label_loss.backward()
        optimizer.step_and_update_lr()

    #     # backward
    #     loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
    #     loss += label_loss
    #     loss.backward()

    #     # update parameters
    #     optimizer.step_and_update_lr()

    #     # note keeping
    #     total_loss += loss.item()

    #     non_pad_mask = gold.ne(Constants.PAD)
    #     n_word = non_pad_mask.sum().item()
    #     n_word_total += n_word
    #     n_word_correct += n_correct

    # loss_per_word = total_loss/n_word_total
    # accuracy = n_word_correct/n_word_total
    # return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    true_labels = []
    preds = []

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos, true_label = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred, pred_label = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct


            pred_label = torch.squeeze(pred_label)
            print(pred_label)
            pred_label = F.sigmoid(pred_label)
            print(pred_label)
            pred_label = torch.where(pred_label>0.5, torch.ones_like(pred_label), pred_label)
            pred_label = torch.where(pred_label<=0.5, torch.zeros_like(pred_label), pred_label)
            print(pred_label)
            this_true_label = true_label.cpu().detach().numpy()
            print(this_true_label)
            raise TypeError
            this_pred = pred_label.cpu().detach().numpy()
            true_labels.append(this_true_label)
            preds.append(this_pred)

    print(roc_auc_score(np.concatenate(true_labels, axis=0), np.concatenate(preds, axis=0)))
    # loss_per_word = total_loss/n_word_total
    # accuracy = n_word_correct/n_word_total
    # return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        # train_loss, train_accu = train_epoch(
        #     model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        # print('- (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
        #       'elapse: {elapse:3.3f} min'.format(
        #           ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
        #           elapse=(time.time()-start)/60))

        start = time.time()
        eval_epoch(model, validation_data, device)
        # valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        # print('- (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
        #         'elapse: {elapse:3.3f} min'.format(
        #             ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
        #             elapse=(time.time()-start)/60))

        # valid_accus += [valid_accu]

        # model_state_dict = model.state_dict()
        # checkpoint = {
        #     'model': model_state_dict,
        #     'settings': opt,
        #     'epoch': epoch_i}

        # if opt.save_model:
        #     if opt.save_mode == 'all':
        #         model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
        #         torch.save(checkpoint, model_name)
        #     elif opt.save_mode == 'best':
        #         model_name = opt.save_model + '.chkpt'
        #         if True:
        #             torch.save(checkpoint, model_name)
        #             print('- [Info] The checkpoint file has been updated.')

        # if log_train_file and log_valid_file:
        #     with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
        #         log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
        #             epoch=epoch_i, loss=train_loss,
        #             ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
        #         log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
        #             epoch=epoch_i, loss=valid_loss,
        #             ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device ,opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        EncoderDecoderDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt'],
            label = data['train']['label']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        EncoderDecoderDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt'],
            label = data['valid']['label']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
        
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
