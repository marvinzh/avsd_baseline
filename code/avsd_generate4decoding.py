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
def generate_response(model, data, batch_indices, vocab,eos=2, maxlen=20, beam=5, penalty=2.0, nbest=1):
    # path for 20-hypos file
    PATH = "/net/callisto/storage1/baiyuu/avsd_system/data/structed_nbest.json"
    hypos = json.load(open(PATH))
    logging.info("lodding hypos file: %s"%PATH)
    #  lambda for P(a|q) model
    lamb= 0.5 

    vocablist = sorted(vocab.keys(), key=lambda s:vocab[s])
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for dialog in data['original']['dialogs']:
            vid = dialog['image_id']
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(dialog['dialog'])}
            result_dialogs.append(pred_dialog)
            for t, qa in enumerate(dialog['dialog']):
                key= "%s_%d" % (vid, t)
                ans_set = hypos[key]["nlist"]
                logging.info("%s: # of hypos in list: %d"%(key, len(ans_set)))
                
                logging.info('%d %s_%d' % (qa_id, vid, t))
                logging.info('QS: ' + qa['question'])
                logging.info('REF: ' + qa['answer'])
                # prepare input data
                start_time = time.time()
                x_batch, h_batch, q_batch, a_batch_in, a_batch_out = \
                        dh.make_batch(data, batch_indices[qa_id])
                qa_id += 1
                # multi modal info.
                x = [torch.from_numpy(x) for x in x_batch]
                    # history info.
                h = [[torch.from_numpy(h) for h in hb] for hb in h_batch]
                    # question info.
                q = [torch.from_numpy(q) for q in q_batch]

                rst_set=[]
                for ans, q2a_logp in ans_set:
                    ans_idx = ans.split()
                    ans_idx = list(map(lambda x:vocab[x], ans_idx))
                    ans_idx = ans_idx + [eos]
                    ans_idx = torch.tensor(ans_idx, dtype=torch.int32)

                    a = [ans_idx]
                    
                    # generate sequences
                    es = model.generate(x, h, a)
                    a2q_logp = calc_logp(model, es, q)

                    final_logp = lamb* q2a_logp + (1. - lamb) * a2q_logp
                    rst.append(
                        (ans, final_logp)
                    )

                rst_set = sorted(rst_set, key=lambda x:x[1])
                pred_dialog['dialog'][t]['answer'] = rst_set[-1][0]
                for ans, logp in rst:
                    logging.info("Answer: %s, logp: %f"%(ans, logp))

                logging.info('ElapsedTime: %f' % (time.time() - start_time))
                logging.info('-----------------------')

    return {'dialogs': result_dialogs}

def calc_logp(model, state, q, sos=2, eos=2, unk=0,):
    assert len(q)==1
   
    q = list(q[0].data.numpy())
    logging.info(q)

    decoder_state = model.response_decoder.initialize(None, state, torch.from_numpy(np.asarray([sos])).cuda())
    a2q_logp = 0.
    for i in q:
        logp = model.response_decoder.predict(decoder_state)
        lp_vec = logp.squeeze().cpu().data.numpy()
        a2q_logp += lp_vec[i]
        decoder_state = model.response_decoder.update(decoder_state, torch.from_numpy(np.asarray([i])).cuda())
    
    return a2q_logp

    # anwsers=[]
    # for ans, q2a_logp in ans_set:
    #     ans_idx = ans.split()
    #     ans_idx = list(map(lambda x:vocab[x], ans_idx))
    #     ans_idx = ans_idx + [eos]

    #     decoder_state = model.response_decoder.initialize(None, state, torch.from_numpy(np.asarray([sos])).cuda())

    #     a2q_logp = 0.
    #     for i in ans_idx:
    #         logp = model.response_decoder.predict(decoder_state)
    #         lp_vec = logp.squeeze().cpu().data.numpy()
    #         a2q_logp += lp_vec[i]
    #         decoder_state = model.response_decoder.update(decoder_state, torch.from_numpy(np.asarray([i])).cuda())
        
    #     final_logp = lamb * q2a_logp + (1. - lamb) * a2q_logp
    #     anwsers.append(
    #         (ans, final_logp)
    #     )

    # anwsers = sorted(anwsers, key=lambda x:x[1])
    # return anwsers



def ranked_beam_search(model, state, ans, sos=2, eos=2, unk=0, minlen=1, beamsize=5, maxlen=20, penalty=2.0, nbest=1):
    '''beam search given answer
    
    Arguments:
        model {[type]} -- [description]
        state {[type]} -- [description]
        ans {list of int} -- [description]
    
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
    decoder_state = model.response_decoder.initialize(None,
                                                    state,
                                                    torch.from_numpy(np.asarray([sos])).cuda())
    
    hyplist = [
        ([], 0., decoder_states),
        ]
    best_state = None
    comp_hyplist = []
    for l in six.moves.range(maxlen):
        new_hyplist = []
        argmin = 0
        
        for out, lp, states in hyplist:
            logp = model.response_decoder.predict(state)

            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)
            if l >= minlen:
                new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                new_st = model.response_decoder.update(states, torch.from_numpy(np.asarray([eos])).cuda())
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state[0] < new_lp:
                    best_state = (new_lp, new_state)

            for o in np.argsort(lp_vec)[::-1]:
                if o == unk or o == eos:  # exclude <unk> and <eos>
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == beamsize:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = model.response_decoder.update(states, torch.from_numpy(np.asarray([o])).cuda())
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = model.response_decoder.update(states,
                                                            torch.from_numpy(np.asarray([o])).cuda())

                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beamsize:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

        hyplist = new_hyplist
    
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
        return maxhyps, best_state[1]
    else:
        return [
            ([], 0)
            ], None

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
 
    logging.info('Loading model params from ' + args.model)
    path = args.model_conf
    with open(path, 'r') as f:
        vocab, train_args = pickle.load(f)
    model = torch.load(args.model+'.pth.tar')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    result = generate_response(model, test_data, test_indices, vocab, 
                               maxlen=args.maxlen, beam=args.beam, 
                               penalty=args.penalty, nbest=args.nbest)
    logging.info('----------------')
    logging.info('wall time = %f' % (time.time() - start_time))
    if args.output:
        logging.info('writing results to ' + args.output)
        json.dump(result, open(args.output, 'w'), indent=4)
    logging.info('done')
