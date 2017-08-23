#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""
import sys
import numpy
import torch
import torch.nn as nn
from . import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )


    def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x1_sentence_s, x1_sentence_e, x1_sentence_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
	x1_sentence = document sentence indices[batch * len_d_sentences]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)

        # Encode document with RNN
        doc_hiddens_wordlevel = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

        '''
        # Add average pooling for document sentence
        print('x1: ' + str(x1))
        print('x1_f: ' + str(x1_f))
        print('x1_mask: ' + str(x1_mask))
        print('x2: ' + str(x2))
        print('x2_mask: ' + str(x2_mask))
        print('x1_sentence: ' + str(x1_sentence))
        print('doc_hiddens_wordlevel type: ' + str(type(doc_hiddens_wordlevel)))
        print('doc_hiddens_wordleve: ' + str(doc_hiddens_wordlevel))
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        print('question_hiddens: ' + str(question_hiddens))
        print('x2_mask: ' + str(x2_mask))
        doc_hidden = []
        for batch_i in range(len(x1_sentence)):
            sent_cur = 0
            doc_hiddens = []
            for sent in x1_sentence[batch_i]:
                sent_pre = sent_cur
                sent_cur = int(sent.data.numpy()[0])
                print('type of sent_cur: ' + str(type(sent_cur)))
                print('sent_cur: ' + str(sent_cur) + '; sent_pre: ' + str(sent_pre))
                print('len of doc_hiddens_wordlevel: ' + str(len(doc_hiddens_wordlevel[batch_i])))
                print('len of x1_mask: ' + str(len(x1_mask[batch_i])))
                doc_merge_weights = layers.uniform_weights_sentence([doc_hiddens_wordlevel[batch_i][sent_pre:sent_cur]], [x1_mask[batch_i][sent_pre:sent_cur]])
                doc_hidden_sentence = layers.weighted_avg_sentence([doc_hiddens_wordlevel[batch_i][sent_pre:sent_cur]], doc_merge_weights)
                doc_hiddens.extend(doc_hidden_sentence)
                doc_hidden_batch = ' '.join(doc_hiddens)
                doc_hidden.append(doc_hidden_batch)

	doc_merge_weights = layers.uniform_weights_sentence(x1_sentence, doc_hiddens_wordlevel, x1_mask)
	doc_hidden = layers.weighted_avg_sentence(x1_sentence, doc_hiddens_wordlevel, doc_merge_weights)
        '''

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)


        def get_doc_sent(doc_word, indices):
            doc_sent = []
            for i in range(doc_word.size(0)):
                tensor_tmp = torch.index_select(doc_word[i], 0, indices[i])
                doc_sent.append(torch.unsqueeze(tensor_tmp, 0))
            return torch.cat(doc_sent, 0)
        doc_hiddens_start = get_doc_sent(doc_hiddens_wordlevel, x1_sentence_s)
        doc_hiddens_end = get_doc_sent(doc_hiddens_wordlevel, x1_sentence_e)
        doc_hiddens_cat = torch.cat((torch.unsqueeze(doc_hiddens_start, 0), torch.unsqueeze(doc_hiddens_end, 0)), 0)
        doc_hiddens = torch.mean(doc_hiddens_cat, 0).squeeze(0)        
        '''
        print('x1_sentence_s: ' + str(x1_sentence_s))
        print('x1_sentence_e: ' + str(x1_sentence_e))
        print('doc_hiddens_start: ' + str(doc_hiddens_start))
        print('doc_hiddens_end: ' + str(doc_hiddens_end))
        print('x1_sentence_mask: ' + str(x1_sentence_mask))
        print('doc_hiddens_wordlevel: ' + str(doc_hiddens_wordlevel))
        print('doc_hiddens_start: ' + str(doc_hiddens_start))
        print('doc_hiddens_end: ' + str(doc_hiddens_end))
        print('doc_hiddens: ' + str(doc_hiddens))
        print('question_hidden: ' + str(question_hidden))
        sys.exit()

        for batch_i in range(x1_sentence.size(0)):
            indices = x1_sentence[batch_i:batch_i+1]
            doc_hiddens[batch_i,:,:].copy_(torch.index_select(doc_hiddens_wordlevel[batch_i:batch_i+1], 1, indices))
        '''


        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_sentence_mask)
        #end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores#, end_scores
