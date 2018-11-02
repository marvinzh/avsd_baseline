#!/usr/bin/env python
"""Multimodal sequence encoder 
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMEncoder(nn.Module):

    def __init__(self, in_size, out_size, enc_psize=[], enc_hsize=[], att_size=128,
                 state_size=100, device="cuda:0", enc_layers=[2,2], mm_att_size=128):
        if len(enc_psize)==0:
            enc_psize = in_size
        if len(enc_hsize)==0:
            enc_hsize = [0] * len(in_size)

        # make links
        super(MMEncoder, self).__init__()
        # memorize sizes
        self.n_inputs = len(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.enc_psize = enc_psize
        self.enc_hsize = enc_hsize
        self.enc_layers = enc_layers
        self.att_size = att_size
        self.state_size = state_size
        self.mm_att_size = mm_att_size
        # encoder
        self.f_lstms = nn.ModuleList()
        self.b_lstms = nn.ModuleList()
        self.emb_x = nn.ModuleList()
        self.device= torch.device(device)
        for m in six.moves.range(len(in_size)):
            self.emb_x.append(nn.Linear(self.in_size[m], self.enc_psize[m]))

            if enc_hsize[m] > 0:
                # create module for stacked bi-LSTM
                self.f_lstms.append(torch.nn.ModuleList())
                self.b_lstms.append(torch.nn.ModuleList())
                # create stacked bi-LSTM for current modality m
                for layer in range(self.enc_layers[m]):
                    if layer == 0:
                        self.f_lstms[m].append(nn.LSTMCell(enc_psize[m], enc_hsize[m]).to(self.device))
                        self.b_lstms[m].append(nn.LSTMCell(enc_psize[m], enc_hsize[m]).to(self.device))
                    else:
                        self.f_lstms[m].append(nn.LSTMCell(enc_hsize[m], enc_hsize[m]).to(self.device))
                        self.b_lstms[m].append(nn.LSTMCell(enc_hsize[m], enc_hsize[m]).to(self.device))

                # self.b_lstms.append(nn.LSTMCell(enc_psize[m], enc_hsize[m]).to(self.device))
        # temporal attention
        self.atV = nn.ModuleList()
        self.atW = nn.ModuleList()
        self.atw = nn.ModuleList()
        self.lgd = nn.ModuleList()
        for m in six.moves.range(len(in_size)):
            enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
            self.atV.append(nn.Linear(enc_hsize_, att_size))
            self.atW.append(nn.Linear(state_size, att_size))
            self.atw.append(nn.Linear(att_size, 1))
            self.lgd.append(nn.Linear(enc_hsize_, out_size))


        # multimodal attention
        self.mm_atts = nn.ModuleList()
        self.qest_att = nn.Linear(128, self.mm_att_size)
        self.mm_att_w = nn.Linear(self.mm_att_size, 1, bias=False)
        for m in six.moves.range(len(in_size)):
            enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
            self.mm_atts.append(nn.Linear(enc_hsize_, self.mm_att_size))
    
        

    # Make an initial state
    def make_initial_state(self, hiddensize):
        return (
            # initial hidden state
            torch.zeros(self.bsize, hiddensize, dtype=torch.float).to(self.device),
            # initial cell state
            torch.zeros(self.bsize, hiddensize, dtype=torch.float).to(self.device),
        )

    # Encoder functions
    def embed_x(self, x_data, m):
        x0 = [x_data[i]
              for i in six.moves.range(len(x_data))]
        return self.emb_x[m](torch.cat(x0, 0).cuda().float())

    # Encoder main
    def encode(self, x):
        h1 = [None] * self.n_inputs
        for m in six.moves.range(self.n_inputs):
            if self.enc_hsize[m] > 0:
                # embedding
                seqlen = len(x[m])
                h0 = self.embed_x(x[m], m)
                # forward path
                fh1 = torch.split(
                        F.dropout(h0, training=self.train), 
                        self.bsize, dim=0)

                hs, cs= self.make_initial_state(self.enc_hsize[m])
                # extend initial hidden state and cell state for stacked LSTM
                hs = [[hs]] * len(self.b_lstms[m])
                cs = [[cs]] * len(self.b_lstms[m])
                h1f = []

                for h in fh1:
                    for level in range(len(self.f_lstms[m])):
                        if level==0:
                            hs_temp, cs_temp = self.f_lstms[m][level](
                                h, 
                                (hs[level][-1], cs[level][-1])
                                )
                        else:
                            hs_temp, cs_temp = self.f_lstms[m][level](
                                hs[level-1][-1],
                                (hs[level][-1],cs[level][-1])
                                )
                        hs[level].append(hs_temp)
                        cs[level].append(cs_temp)
                    # fstate = self.f_lstms[m](h,fstate)
                    h1f.append(hs[-1][-1])

                # backward path
                bh1 = torch.split(
                        F.dropout(h0, training=self.train),
                        self.bsize, dim=0)

                hs, cs = self.make_initial_state(self.enc_hsize[m])
                 # extend initial hidden state and cell state for stacked LSTM
                hs = [[hs]] * len(self.b_lstms[m])
                cs = [[cs]] * len(self.b_lstms[m])
                h1b = []
                for h in reversed(bh1):
                    for level in range(len(self.b_lstms[m])):
                        if level == 0:
                            hs_temp, cs_temp = self.b_lstms[m][level](
                                h,
                                (hs[level][-1], cs[level][-1])
                                )
                        else:
                            hs_temp, cs_temp = self.b_lstms[m][level](
                                hs[level-1][-1],
                                (hs[level][-1], cs[level][-1])
                                )
                        hs[level].append(hs_temp)
                        cs[level].append(cs_temp)

                    # bstate = self.b_lstms[m](h, bstate)
                    h1b.insert(0, hs[-1][-1])

                # concatenation
                h1[m] = torch.cat([torch.cat((f, b), 1)
                                   for f, b in six.moves.zip(h1f, h1b)], 0)
            else:
                # embedding only
                h1[m] = torch.tanh(self.embed_x(x[m], m))
        return h1

    # Attention
    def attention(self, h, vh, s):
        c = [None] * self.n_inputs

        for m in six.moves.range(self.n_inputs):
            bsize = self.bsize
            seqlen = h[m].data.shape[0] / bsize
            csize = h[m].data.shape[1]
            asize = self.att_size

            ws = self.atW[m](s)
            vh_m = vh[m].view(seqlen, bsize, asize)
            e1 = vh_m + ws.expand_as(vh_m)
            e1 = e1.view(seqlen * bsize, asize)
            e = torch.exp(self.atw[m](torch.tanh(e1)))
            e = e.view(seqlen, bsize)
            esum = e.sum(0)
            e = e / esum.expand_as(e)
            h_m = h[m].view(seqlen, bsize, csize)
            h_m = h_m.permute(2,0,1)
            c_m = h_m * e.expand_as(h_m)
            c_m = c_m.permute(1,2,0)
            c[m] = c_m.mean(0)
        return c

    def mm_attention(self, g_q, c):
        wg= self.qest_att(g_q)
        vs=[]
        for i in range(self.n_inputs):
             vs.append(
                 self.mm_atts[i](c[i]) + wg
                 )

        # each elems in vs (B, atten_size)
        for i in range(self.n_inputs):
            vs[i] = self.mm_att_w(torch.tanh(vs[i]))
        
        # each elems in vs (B, 1)
        vs = torch.cat(vs,dim=1)
        
        #  (B, # of modality)
        beta = torch.softmax(vs, dim=1)
        
        # nsize = self.n_inputs
        # bsize = self.bsize
        # asize = self.att_size
        # out_size = self.out_size


        # vc_n = []
        # for m in six.moves.range(nsize):
        #     vc_n.append(self.mm_atts[m](c[m]))
            
        # ws = self.qest_att(g_q)                      # ws=(bsize, asize)
        # vc_n = torch.cat(vc_n, dim=0)
        # vc_n = vc_n.view(nsize, bsize, asize)
        
        # e1 = vc_n + ws.expand_as(vc_n)             #(nsize,bsize,asize)
        # e1 = e1.view(nsize * bsize, asize)

        # e = self.mm_att_w(torch.tanh(e1))   
        
        # e = e.view(nsize, bsize)    
        # beta = F.softmax(e, dim=0)   

        beta = beta.permute(1,0)
        # (batchsize, #modality)
        return beta
    
    def att_modality_fusion(self, c, beta):
        # assert beta.shape[1] == self.n_inputs

        # beta = beta.permute(1,0)
        # beta: (# of modality, B)

        # g = 0.
        # for m in range(self.n_inputs):
            # g += beta[m].view(-1,1) * self.lgd[m](F.dropout(c[m]))
        # return g
        nsize = self.n_inputs
        bsize = self.bsize
        asize = self.att_size
        out_size = self.out_size

        d_n = [self.lgd[m](F.dropout(c[m])) for m in six.moves.range(self.n_inputs)]
        
        d_n = torch.cat(d_n).view(nsize, bsize, out_size) 
        d_n = d_n.permute(2, 0, 1)                  #(out_size, nsize, bsize)
        
        g_n = d_n * beta.expand_as(d_n)
        g_n = g_n.permute(1, 2, 0)                  #(nsize, bsize, out_size)
        g_n = g_n.sum(0)
        return g_n

    # Simple modality fusion
    def simple_modality_fusion(self, c, s):

        g = 0.
        for m in six.moves.range(self.n_inputs):
            g += self.lgd[m](F.dropout(c[m]))
        return g

    # forward propagation routine
    def __call__(self, s, x, train=True):
        '''multimodal encoder main
        
        Arguments:
            s {[type]} -- question encoding
            x {[type]} -- raw multi-modal feature
        
        Keyword Arguments:
            train {bool} -- [description] (default: {True})
        
        Returns:
            [type] -- [description]
        '''

        self.bsize = x[0][0].shape[0]
        
        h1 = self.encode(x)
        vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]

        # attention
        c = self.attention(h1, vh1, s)

        beta = self.mm_attention(s, c)
        g = self.att_modality_fusion(c, beta)
        # g = self.simple_modality_fusion(c, s)
        
        return torch.tanh(g)

#!/usr/bin/env python
# """Multimodal sequence encoder 
#    Copyright 2016 Mitsubishi Electric Research Labs
#    WANG official attention, using different att_V[m] 
# """

# import math
# import numpy as np
# import six
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import logging

# class MMEncoder(nn.Module):

#     def __init__(self, in_size, out_size, enc_psize=[], enc_hsize=[], att_size=100,
#                  state_size=100):
#         if len(enc_psize)==0:
#             enc_psize = in_size
#         if len(enc_hsize)==0:
#             enc_hsize = [0] * len(in_size)
        
#         #print('WANG test values-----------------')
#         #print('enc_psize', enc_psize)  #[512, 512, 64]
#         #print('enc_hsize', enc_hsize)  #[0, 0, 0]
#         #print('in_size', in_size)      #[2048, 2048, 128]
#         #print('out_size', out_size)    #256
#         #print('---------------------------------')
        
#         # make links
#         super(MMEncoder, self).__init__()
#         # memorize sizes
#         self.n_inputs = len(in_size)
#         self.in_size = in_size
#         self.out_size = out_size
#         self.enc_psize = enc_psize
#         self.enc_hsize = enc_hsize
#         self.att_size = att_size
#         self.state_size = state_size
#         # encoder
#         self.l1f_x = nn.ModuleList()    # forward path
#         self.l1f_h = nn.ModuleList()
#         self.l1b_x = nn.ModuleList()    # backward path
#         self.l1b_h = nn.ModuleList()
#         self.emb_x = nn.ModuleList()
#         for m in six.moves.range(len(in_size)):
#             self.emb_x.append(nn.Linear(self.in_size[m], self.enc_psize[m]))
#             if enc_hsize[m] > 0:
#                 self.l1f_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
#                 self.l1f_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias=False))
#                 self.l1b_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
#                 self.l1b_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias=False))
        
#         # temporal attention
#         self.atV = nn.ModuleList()
#         self.atW = nn.ModuleList()
#         self.atw = nn.ModuleList()
#         self.lgd = nn.ModuleList()
#         for m in six.moves.range(len(in_size)):
#             enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
#             self.atV.append(nn.Linear(enc_hsize_, att_size))
#             self.atW.append(nn.Linear(state_size, att_size))
#             self.atw.append(nn.Linear(att_size, 1))
#             self.lgd.append(nn.Linear(enc_hsize_, out_size))
        
#         # [WANG] multimodal attention
#         self.f_att_V = nn.ModuleList()
#         for m in six.moves.range(len(in_size)):
#             enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m] #[512,512,64]
#             self.f_att_V.append(nn.Linear(enc_hsize_, att_size))
#         self.f_att_W = nn.Linear(state_size, att_size)
#         self.f_att_w = nn.Linear(att_size, 1, bias=False)
    
#     # Make an initial state
#     def make_initial_state(self, hiddensize):
#         return {name: torch.zeros(self.bsize, hiddensize, dtype=torch.float)
#                 for name in ('c1', 'h1')}

#     # Encoder functions
#     def embed_x(self, x_data, m):
#         x0 = [x_data[i]
#               for i in six.moves.range(len(x_data))]
#         return self.emb_x[m](torch.cat(x0, 0).cuda().float())

#     def forward_one_step(self, x, s, m):
#         x_new = x + self.l1f_h[m](s['h1'].cuda())
#         x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
#         x_list = list(x_list)
#         c1 = torch.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
#         h1 = torch.tanh(c1) * F.sigmoid(x_list[3])
#         return {'c1': c1, 'h1': h1}

#     def backward_one_step(self, x, s, m):
#         x_new = x + self.l1b_h[m](s['h1'].cuda())
#         x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
#         x_list = list(x_list)
#         c1 = torch.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
#         h1 = torch.tanh(c1) * F.sigmoid(x_list[3])
#         return {'c1': c1, 'h1': h1}

#     # Encoder main
#     def encode(self, x):
#         h1 = [None] * self.n_inputs
#         for m in six.moves.range(self.n_inputs):
#             # self.emb_x=self.__dict__['emb_x%d' % m]
#             if self.enc_hsize[m] > 0:
#                 # embedding
#                 seqlen = len(x[m])  # 126 x=[tensor(126, 64, 2048)] 
#                 h0 = self.embed_x(x[m], m) #  (seqlen*bsize, 2048)>(seqlen*bsize, 512)                   [2048,2048,128]>[512, 512, 64]

#                 # forward path
#                 fh1 = torch.split(self.l1f_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)  # seqlen * (bsise, 4* hsize)
#                 fstate = self.make_initial_state(self.enc_hsize[m])   #{'c1'=(bsize,hsize)}
#                 h1f = []
#                 for h in fh1:
#                     fstate = self.forward_one_step(h, fstate, m)  #return new {c,h}
#                     h1f.append(fstate['h1']) #each seq will have different h    seqlen * [(bsize,hsize)]
#                 # backward path
#                 bh1 = torch.split(self.l1b_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)   #
#                 bstate = self.make_initial_state(self.enc_hsize[m])
#                 h1b = []
#                 for h in reversed(bh1):
#                     bstate = self.backward_one_step(h, bstate, m)
#                     h1b.insert(0, bstate['h1'])
#                 # concatenation
#                 h1[m] = torch.cat([torch.cat((f, b), 1)                     # [seqlen*bsize,2*hsize]
#                                    for f, b in six.moves.zip(h1f, h1b)], 0) # cat the forward and back h for same seq
#             else:
#                 # embedding only
#                 h1[m] = torch.tanh(self.embed_x(x[m], m))   #(seqlen*bsize, 2048)>(seqlen*bsize, 512)  
#         return h1

#     # Attention
#     def attention(self, h, vh, s):
#         c = [None] * self.n_inputs

#         for m in six.moves.range(self.n_inputs):
#             bsize = self.bsize                   # 64
#             seqlen = h[m].data.shape[0] / bsize  #seqlen*bsize/bsize
#             csize = h[m].data.shape[1]
#             asize = self.att_size               #128

#             ws = self.atW[m](s)                 # state_size128>att 128 ws=(bsize,att_size)
#             vh_m = vh[m].view(seqlen, bsize, asize)     #vh=(seqlen,bszie,att_size)
#             e1 = vh_m + ws.expand_as(vh_m)   
#             e1 = e1.view(seqlen * bsize, asize)
#             e = torch.exp(self.atw[m](torch.tanh(e1)))
#             e = e.view(seqlen, bsize)
#             esum = e.sum(0)
#             e = e / esum.expand_as(e)            #e = (seqlen,bsize) 
#             #logging.info('test0 temporal e is: {}'.format(e))
#             h_m = h[m].view(seqlen, bsize, csize)
#             h_m = h_m.permute(2,0,1)          #h_m=(csize,seqlen,bsize)
#             c_m = h_m * e.expand_as(h_m)
#             c_m = c_m.permute(1,2,0)   # c_m=(seqlen,bsize,csize)
#             c[m] = c_m.mean(0)   # (bsize,csize)
#         return c

#     # Simple modality fusion
#     def simple_modality_fusion(self, c, s):
#         g = 0.
#         for m in six.moves.range(self.n_inputs):
#             g += self.lgd[m](F.dropout(c[m]))
#         return g
        
#     #[WANG] 10.23 multimodal attention same as the ref, bad results
#     def att_modality_fusion1(self, c, d, s):
#         nsize = self.n_inputs
#         bsize = self.bsize
#         asize = self.att_size
#         out_size = self.out_size
#         vc_n = []
#         for m in six.moves.range(nsize):
#             csize = c[m].data.shape[1]
#             vc_n.append(self.f_att_V[m](c[m]))
#         ws = self.f_att_W(s)                       # ws=(bsize, asize)
#         vc_n = torch.cat(vc_n, dim=0)
#         vc_n = vc_n.view(nsize, bsize, asize)
#         e1 = vc_n + ws.expand_as(vc_n)             #(nsize,bsize,asize)
#         e1 = e1.view(nsize * bsize, asize)    
#         e = torch.exp(self.f_att_w(torch.tanh(e1)))    
#         e = e.view(nsize, bsize)    
#         esum = e.sum(0)                            #(bsize)
#         e = e / esum.expand_as(e)                  #(nsize, bsize)
#         d_n = torch.cat(d).view(nsize, bsize, out_size) 
#         d_n = d_n.permute(2, 0, 1)           #(out_size, nsize, bsize)
#         g_n = d_n * e.expand_as(d_n)
#         g_n = g_n.permute(1, 2, 0)           #(nsize, bsize, out_size)
#         g_n = g_n.sum(0)
        
#         return g_n
    
#     #[WANG] 10.24 multimodal attention same as the ref, good results
#     def att_modality_fusion(self, c, d, s):
#         nsize = self.n_inputs
#         bsize = self.bsize
#         asize = self.att_size
#         out_size = self.out_size


#         vc_n = []
#         for m in six.moves.range(nsize):
#             vc_n.append(self.f_att_V[m](c[m]))
            
#         ws = self.f_att_W(s)                       # ws=(bsize, asize)
#         vc_n = torch.cat(vc_n, dim=0)
#         vc_n = vc_n.view(nsize, bsize, asize)
        
#         e1 = vc_n + ws.expand_as(vc_n)             #(nsize,bsize,asize)
#         e1 = e1.view(nsize * bsize, asize)

#         e = self.f_att_w(torch.tanh(e1))   
        
#         e = e.view(nsize, bsize)    
#         e = F.softmax(e,dim=0)                      #(nsize,bszie)
        
#         d_n = torch.cat(d).view(nsize, bsize, out_size) 
#         d_n = d_n.permute(2, 0, 1)                  #(out_size, nsize, bsize)
        
#         g_n = d_n * e.expand_as(d_n)
#         g_n = g_n.permute(1, 2, 0)                  #(nsize, bsize, out_size)
#         g_n = g_n.sum(0)
        
#         return g_n
    
#     # forward propagation routine
#     def __call__(self, s, x, train=True):
       
#         self.bsize = x[0][0].shape[0]   # 64 x=[tensor(126, 64, 2048)]
#         h1 = self.encode(x)             # hsize=0: [seqlen*bsize,psize]      hsize>0: [seqlen*bsize,2*hsize]
#         vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]  # vh1[0] = (seqlen*bsize, 128)
#         # attention
#         c = self.attention(h1, vh1, s)  #c=[(bsize,2*hsize)]*n_inputs
     
#         d = [self.lgd[m](F.dropout(c[m])) for m in six.moves.range(self.n_inputs)] # 2 * enc_hsize[m] > out_size 256 #d = nsize * [tensor(bsize,256)] 
#         g = self.att_modality_fusion(c, d, s)# g = self.simple_modality_fusion(c, s)
#         return torch.tanh(g)    # return g(av) part

