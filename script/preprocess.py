#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import sys
sys.path.append('.')
import argparse
import os
try:
	import ujson as json
except ImportError:
	import json
import time

from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from functools import partial
from spacy_tokenizer import SpacyTokenizer

import nltk

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None
ANNTOTORS = {'lemma', 'pos', 'ner'}


def init():
	global TOK
	TOK = SpacyTokenizer(annotators=ANNTOTORS)
	Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
	"""Call the global process tokenizer on the input text."""
	global TOK
	tokens = TOK.tokenize(text)
	output = {
		'words': tokens.words(),
		'chars': tokens.chars(),
		'offsets': tokens.offsets(),
		'pos': tokens.pos(),
		'lemma': tokens.lemmas(),
		'ner': tokens.entities(),
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
			  'contexts': [], 'qid2cid': [], 'sentences': []}
	for article in data:
		for paragraph in article['paragraphs']:
			output['contexts'].append(paragraph['context'])

			context=paragraph['context']
			sentenceList=nltk.sent_tokenize(context)
			output['sentences'].append(sentenceList)

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


def process_dataset(data, tokenizer, workers=None):
	"""Iterate processing (tokenize, parse, etc) dataset multithreaded."""
	make_pool = partial(Pool, workers, initializer=init)

	workers = make_pool(initargs=())
	q_tokens = workers.map(tokenize, data['questions'])
	workers.close()
	workers.join()

	workers = make_pool(initargs=())
	c_tokens = workers.map(tokenize, data['contexts'])
	workers.close()
	workers.join()

	#print(len(data['sentences']))
	#print(len(c_tokens))

#	workers = make_pool(initargs=())
#	s_tokensList = []
#	for sent in data['sentences']:
#		s_tokens = workers.map(tokenize, sent)
#		s_tokensList.append(s_tokens)
#	workers.close()
#	workers.join()

	for idx in range(len(data['qids'])):
		question = q_tokens[idx]['words']
		question_char = q_tokens[idx]['chars']
		qlemma = q_tokens[idx]['lemma']
		qpos = q_tokens[idx]['pos']
		qner = q_tokens[idx]['ner']

		document = c_tokens[data['qid2cid'][idx]]['words']
		document_char = c_tokens[data['qid2cid'][idx]]['chars']
		offsets = c_tokens[data['qid2cid'][idx]]['offsets']
		clemma = c_tokens[data['qid2cid'][idx]]['lemma']
		cpos = c_tokens[data['qid2cid'][idx]]['pos']
		cner = c_tokens[data['qid2cid'][idx]]['ner']
		
		ans_tokens = []
		if len(data['answers']) > 0:
			for ans in data['answers'][idx]:
				found = find_answer(offsets,
									ans['answer_start'],
									ans['answer_start'] + len(ans['text']))
				if found:
					ans_tokens.append(found)

		senIdxList = []
		senBeginWordIndex =0
		senEndCharIndex = 0
		#for sen in range(len(s_tokensList[data['qid2cid'][idx]])):
		#	senEndCharIndex += len(' '.join(s_tokensList[data['qid2cid'][idx]][sen]['words']))
		sentList=data['sentences'][data['qid2cid'][idx]]
		for j in range(len(sentList)):
			sent = sentList[j]
			if(j!=0):
				senEndCharIndex+=1
			senEndCharIndex +=len(sent)

			for i in range(len(offsets)):
				if(i!=len(offsets)-1):
					charNextBeginIndex = offsets[i+1][0]
					if(senEndCharIndex-1 < charNextBeginIndex):
						senIdxList.append([senBeginWordIndex,i+1])
						senBeginWordIndex = i+1
						break
				else:
					senIdxList.append([senBeginWordIndex,i+1])
					break
				

		ansSenIdxList = []
		for i in range(len(ans_tokens)):
			ans_begin_idx, ans_end_idx = ans_tokens[i]
			for j in range(len(senIdxList)):
				senBeginWordIndex, senEndIndex = senIdxList[j]
				if(senBeginWordIndex <= ans_begin_idx and senEndIndex >= ans_end_idx):
					ansSenIdxList.append(j)

#		if(len(offsets)==senIdxList[-1][1]):
#			print(document)
#			print(ans_tokens)
#			print(senIdxList)
#			print(ansSenIdxList)

		yield {
			'id': data['qids'][idx],
			'question': question,
			'question_char': question_char,
			'document': document,
			'document_char': document_char,
			'offsets': offsets,
			'answers': ans_tokens,
			'qlemma': qlemma,
			'qpos': qpos,
			'qner': qner,
			'clemma': clemma,
			'cpos': cpos,
			'cner': cner,
			'document_sentence': senIdxList,
			'sentence_answers': ansSenIdxList,
		}


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--tokenizer', type=str, default='spacy')
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
	for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
		f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
