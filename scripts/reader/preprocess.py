#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from drqa import tokenizers

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
	'sent_offsets': tokens.sent_offsets(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]


def find_answer_sentence(sent_offsets, begin_offset):
    """Match token offsets with the sentence begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(sent_offsets) if begin_offset >= tok[0] and begin_offset <= tok[1]]
    #end = [i for i, tok in enumerate(sent_offsets) if tok[1] == end_offset]
    #assert(len(start) <= 1)
    #assert(len(end) <= 1)
    #if len(start) == 1:# and len(end) == 1:
    #    return start[0]#, end[0]
    if len(start) > 0: 
        return start[0]


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    #debug
    # print(type(data))
    # print(data['contexts'])
    c_tokens = workers.map(tokenize, data['contexts'])
    #c_tokens = tokenize(data['contexts'])
    #
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    #debug
    #print('len of data[qids]: ' + str(len(data['qids'])))

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        sent_offsets = c_tokens[data['qid2cid'][idx]]['sent_offsets']#add jyu
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        ans_tokens = []
	# debug
        #print('contexts: ' + str(len(str(data['contexts']))))
        #print('document: ' + str(document))
        #document_sentences = c_tokens[data['qid2cid'][idx]]['sentences']
        #print('document_sentences: ' + document_sentences)
        #sys.exit()

        sent_offsets_distict = []#torch.IntTensor(1) 
        sent_offsets_distict.append(sent_offsets[0])
        for sent_offset in sent_offsets:
            #print(str(sent_offset))
            #print(str(sent_offsets_distict[-1]))
            if sent_offsets_distict[-1][0] < sent_offset[0]:
               sent_offsets_distict.append(sent_offset)
	#sent_offsets_distict = torch.from_numpy(sent_offsets_distict)
        sent_offsets_distict_tensor = torch.from_numpy(np.asarray(sent_offsets_distict))
        #print(type(offsets))
        #print(type(sent_offsets_distict_tensor))
        #print(type(sent_offsets_distict))
        #sent_offsets_distict_tensor =  torch.IntTensor(sent_offsets_distict_tensor)

        sent_index_offsets = []
        sent_index_offset_cur = []
        sent_index_offset_cur.append(0)
        for i in range(len(sent_offsets)-1):
            if sent_offsets[i] != sent_offsets[i+1]:
                sent_index_offset_cur.append(i)
                sent_index_offsets.append(sent_index_offset_cur)
                sent_index_offset_cur = []
                sent_index_offset_cur.append(i+1)
        sent_index_offset_cur.append(len(sent_offsets) - 1)
        sent_index_offsets.append(sent_index_offset_cur)

        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer_sentence(sent_offsets_distict_tensor, ans['answer_start'])
                '''
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                '''
                if found:
                    ans_tokens.append(found)
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            #'sent_offsets_duplicates':sent_offsets,
            'sent_offsets':sent_index_offsets,#add for change into sentence level jyu
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='corenlp')
args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
