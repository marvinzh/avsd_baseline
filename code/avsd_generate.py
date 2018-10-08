#!/usr/bin/env python
"""Scene-aware Dialog Generation
   Copyright 2018 Mitsubishi Electric Research Labs
"""

import argparse
import logging
import math
import sys
import time
import os
import copy
import pickle
import json

import numpy as np
import six

import torch
import torch.nn as nn
import data_handler as dh


# Evaluation routine
def generate_response(models, data, batch_indices, vocab, maxlen=20, beam=5, penalty=2.0, nbest=1):
    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogs = []
    for model in models:
        model.eval()
    with torch.no_grad():
        qa_id = 0
        for dialog in data['original']['dialogs']:
            vid = dialog['image_id']
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(dialog['dialog'])}
            result_dialogs.append(pred_dialog)
            for t, qa in enumerate(dialog['dialog']):
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + qa['question'])
                logging.info('REF: ' + qa['answer'])
                # prepare input data
                start_time = time.time()
                x_batch, h_batch, q_batch, a_batch_in, a_batch_out = \
                    dh.make_batch(data, batch_indices[qa_id])
                qa_id += 1
                x = [torch.from_numpy(x) for x in x_batch]
                h = [[torch.from_numpy(h) for h in hb] for hb in h_batch]
                q = [torch.from_numpy(q) for q in q_batch]
                # generate sequences
                encoder_states=[]
                for model in models:
                    es = model.generate(x, h, q)
                    encoder_states.append(es)

                # pred_out, _ = model.generate(x, h, q, maxlen=maxlen, 
                #                         beam=beam, penalty=penalty, nbest=nbest)
                pred_out, _ = beam_search(models,encoder_states,maxlen=maxlen, beamsize=beam, penalty=penalty, nbest=nbest)
                for n in six.moves.range(min(nbest, len(pred_out))):
                    pred = pred_out[n]
                    hypstr = ' '.join([vocablist[w] for w in pred[0]])
                    logging.info('HYP[%d]: %s  ( %f )' % (n + 1, hypstr, pred[1]))
                    if n==0:
                        pred_dialog['dialog'][t]['answer'] = hypstr
                logging.info('ElapsedTime: %f' % (time.time() - start_time))
                logging.info('-----------------------')

    return {'dialogs': result_dialogs}


def beam_search(models, ss,sos=2, eos=2, unk=0, minlen=1, beamsize=5, maxlen=20, penalty=2.0, nbest=1):
    '''beam search
    
    Arguments:
        models {list of models} -- models for ensemble
        ss {list of states} -- initial states for each model
    
    Keyword Arguments:
        sos {int} -- [description] (default: {2})
        eos {int} -- [description] (default: {2})
        unk {int} -- [description] (default: {0})
        minlen {int} -- [description] (default: {1})
        beamsize {int} -- [description] (default: {5})
        maxlen {int} -- [description] (default: {20})
        penalty {float} -- [description] (default: {2.0})
        nbest {int} -- [description] (default: {1})
    
    Returns:
        [type] -- [description]
    '''
    num_models=len(models)
    decoder_states=[]
    for i, model in enumerate(models):
        decoder_state = model.response_decoder.initialize(None, ss[i], torch.from_numpy(np.asarray([sos])).cuda())
        decoder_states.append(decoder_state)
    
    hyplist = [
        ([], 0., decoder_states),
        ]
    best_state = None
    comp_hyplist = []
    for l in six.moves.range(maxlen):
        new_hyplist = []
        argmin = 0
        
        for out, lp, states in hyplist:

            logp = models[-1].response_decoder.predict(states[-1])
            for i, model in enumerate(models[:-1]):
                logp += model.response_decoder.predict(states[i])
            logp = logp/num_models

            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if l >= minlen:
                new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                new_states =[]
                for i, model in enumerate(models):
                    new_st = model.response_decoder.update(states[i],
                                                        torch.from_numpy(np.asarray([eos])).cuda()
                                                        )
                    new_states.append(new_st)
                
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state[0] < new_lp:
                    best_state = (new_lp, new_states)

            for o in np.argsort(lp_vec)[::-1]:
                if o == unk or o == eos:  # exclude <unk> and <eos>
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == beamsize:
                    if new_hyplist[argmin][1] < new_lp:
                        new_states=[]
                        for i, model in enumerate(models):
                            new_st = model.response_decoder.update(states[i],
                                                              torch.from_numpy(np.asarray([o])).cuda())
                            new_states.append(new_st)

                        new_hyplist[argmin] = (out + [o], new_lp, new_states)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_states=[]
                    for i, model in enumerate(models):
                        new_st = model.response_decoder.update(states[i],
                                                             torch.from_numpy(np.asarray([o])).cuda())
                        new_states.append(new_st)

                    new_hyplist.append((out + [o], new_lp, new_states))
                    if len(new_hyplist) == beamsize:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

        hyplist = new_hyplist
    
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state[1]
    else:
        return [([], 0)], None

##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--test-path', default='', type=str,
                        help='Path to test feature files')
    parser.add_argument('--test-set', default='', type=str,
                        help='Filename of test data')
    parser.add_argument('--model-conf', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--maxlen', default=30, type=int,
                        help='Max-length of output sequence')
    parser.add_argument('--beam', default=3, type=int,
                        help='Beam width')
    parser.add_argument('--penalty', default=2.0, type=float,
                        help='Insertion penalty')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Number of n-best hypotheses')
    parser.add_argument('--output', '-o', default='', type=str,
                        help='Output generated responses in a json file')
    parser.add_argument('--verbose', '-v', default=0, type=int,
                        help='verbose level')

    args = parser.parse_args()

    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
 
    args.model=[
        "/net/callisto/storage1/baiyuu/avsd_system/exp/avsd_i3d_rgb-i3d_flow_Adam_ep512-512_eh0-0_dp128_dh128_att128_bs64_seed1/avsd_model_1",
        "/net/callisto/storage1/baiyuu/avsd_system/exp/avsd_i3d_rgb-i3d_flow_Adam_ep512-512_eh0-0_dp128_dh128_att128_bs64_seed1/avsd_model_2",
        "/net/callisto/storage1/baiyuu/avsd_system/exp/avsd_i3d_rgb-i3d_flow_Adam_ep512-512_eh0-0_dp128_dh128_att128_bs64_seed1/avsd_model_3",
    ]
    path = args.model_conf
    with open(path, 'r') as f:
        vocab, train_args = pickle.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models=[]
    for model_path in args.model:
        logging.info('Loading model params from ' + model_path+'.pth.tar')
        model = torch.load(model_path+'.pth.tar')
        model.to(device)
        models.append(model)
        
    if train_args.dictmap != '':
        dictmap = json.load(open(train_args.dictmap, 'r'))
    else:
        dictmap = None
    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # prepare test data
    logging.info('Loading test data from ' + args.test_set)
    test_data = dh.load(train_args.fea_type, args.test_path, args.test_set,
                        vocab=vocab, dictmap=dictmap, 
                        include_caption=train_args.include_caption)
    test_indices, test_samples = dh.make_batch_indices(test_data, 1)
    logging.info('#test sample = %d' % test_samples)
    # generate sentences
    logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    result = generate_response(models, test_data, test_indices, vocab, 
                               maxlen=args.maxlen, beam=args.beam, 
                               penalty=args.penalty, nbest=args.nbest)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
