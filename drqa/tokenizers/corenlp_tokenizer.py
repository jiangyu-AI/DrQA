#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Simple wrapper around the Stanford CoreNLP pipeline.

Serves commands to a java subprocess running the jar. Requires java 8.
"""

import copy
import json
import pexpect

from .tokenizer import Tokens, Tokenizer
from . import DEFAULTS


class CoreNLPTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        self.classpath = (kwargs.get('classpath') or
                          DEFAULTS['corenlp_classpath'])
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        data = []
        #debug_jy
        '''
        print('len of ouput: (should be # of sentences)' + str(len(output['sentences'])))
        print('sentences: ' + str(output['sentences']))
        sys.exit()
        print('clean_text: ' + clean_text[:53] + '\n' + clean_text[54:127])
        tokens_sentences = output['sentences']
        #print('tokens_sentences: ' + str(tokens_sentences[:2]))
        for tokens_sentence in tokens_sentences:
            print('tokens_sentence: ' + str(tokens_sentence))

        pre = 0
        for i in range(len(sentence_lens)):
            sentence_lens[i] = sentence_lens[i] + pre
            pre = sentence_lens[i]

        index_sent_lens = 0
        index_debug = sentence_lens[0]
        print('index_debug' + str(index_debug))
        print('len of tokens' + str(len(tokens)))
        
        '''

        #debug
        #print('len of output: ' + str(len(output)))
        #print(str(output['sentences'][0]))
        #print(str(output['sentences'][1]))
        sentences = [s for s in output['sentences']]
        sentenceOffsetBegin_list = []
        sentenceOffsetEnd_list = []
        for index_sentence in range(len(sentences)):
            tokens = [t for t in sentences[index_sentence]['tokens']]
            sentenceOffsetBegin_list.append(tokens[0]['characterOffsetBegin'])
            sentenceOffsetEnd_list.append(tokens[-1]['characterOffsetEnd'])

        tokens = [t for s in output['sentences'] for t in s['tokens']]
        index_sent = 0
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']

            #debug
            #print(text[start_ws: end_ws])
            #print('start_ws: ' + str(start_ws) + ' end_ws: ' + str(end_ws))
            #debug
            if tokens[i]['characterOffsetBegin'] > sentenceOffsetEnd_list[index_sent]:
                index_sent = index_sent + 1
            #print(str(index_sent))
            sentenceOffsetBegin = sentenceOffsetBegin_list[index_sent]
            sentenceOffsetEnd = sentenceOffsetEnd_list[index_sent]

            data.append((
                self._convert(tokens[i]['word']),
                text[start_ws: end_ws],
                (tokens[i]['characterOffsetBegin'],
                 tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None),
                (sentenceOffsetBegin,
                 sentenceOffsetEnd),
            ))
        #debug
        #print('data: ' + str(data))

        return Tokens(data, self.annotators)
