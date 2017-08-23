#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])

    sent_index_s = torch.LongTensor([sent[0] for sent in ex['sent_offsets']])
    sent_index_e = torch.LongTensor([sent[1] for sent in ex['sent_offsets']])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id'], sent_index 

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0])
        #end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a for a in ex['answers']]
        #end = [a[1] for a in ex['answers']]


    return document, features, question, sent_index_s, sent_index_e, start, ex['id']

def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 1#2
    NUM_EXTRA = 1
    NUM_SENT_INDEX = 2

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    sent_indexes_s = [ex[3] for ex in batch]
    sent_indexes_e = [ex[4] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    # Batch sentence indexes  added jyu
    max_length = max([s.size(0) for s in sent_indexes_s])
    x1_sent_s = torch.LongTensor(len(sent_indexes_s), max_length).zero_() 
    x1_sent_e = torch.LongTensor(len(sent_indexes_e), max_length).zero_()
    x1_sent_mask = torch.ByteTensor(len(sent_indexes_s), max_length).fill_(1)
    for i, s in enumerate(sent_indexes_s):
        x1_sent_s[i, :s.size(0)].copy_(s)
        x1_sent_mask[i, :s.size(0)].fill_(0)
    for i, s in enumerate(sent_indexes_e):
        x1_sent_e[i, :s.size(0)].copy_(s)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_SENT_INDEX:
        return x1, x1_f, x1_mask, x2, x2_mask, x1_sent_s, x1_sent_e, x1_sent_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS + NUM_SENT_INDEX:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][5]):
            y_s = torch.cat([ex[5] for ex in batch])
            #y_e = torch.cat([ex[6] for ex in batch])
        else:
            y_s = [ex[5] for ex in batch]
            #y_e = [ex[6] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, x1_sent_s, x1_sent_e, x1_sent_mask, y_s, ids
