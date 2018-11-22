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

#import nltk
import spacy

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None
ANNTOTORS = {'lemma', 'pos', 'ner'}
nlp = spacy.load('en')


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
#	nlp=spacy.load('en')

	with open(path) as f:
		data = json.load(f)['data']
	output = {'qids': [], 'questions': [], 'answers': [],
			  'contexts': [], 'qid2cid': [], 'sentences': []}
	for article in data:
		for paragraph in article['paragraphs']:
			output['contexts'].append(paragraph['context'])

			context=paragraph['context'].replace('\n', ' ')
			nlp_context=nlp(context)
			sentenceList = [sent for sent in nlp_context.sents]
			#sentenceTokenIndexList = [[len(sent),] for sent in nlp_context.sents]
			#sentenceList=nltk.sent_tokenize(context)
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
	senBoundError = open("senBoundError_"+args.split+".txt",'w')
	senBoundErrorDict = {}
#	nlp = spacy.load('en')
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
	
	emptySenError = 0
	tokenCountError =0 
	ansSenError = 0
	ansOverlapCount = 0
	ansBESameCount = 0
	wholeAnswerCount = 0

	emptySenCaseList = []
	for idx in range(len(data['qids'])):
		time0 = time.time()

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
		ans_text = []
		if len(data['answers']) > 0:
			for ans in data['answers'][idx]:
				found = find_answer(offsets,
									ans['answer_start'],
									ans['answer_start'] + len(ans['text']))
				if found:
					ans_tokens.append(found)
					ans_text.append(ans['text'])

		time1= time.time()
		# MAKE SENTENCE INDEX LIST [word begin index, word end index + 1]
		senIdxList = []
		senBeginWordIndex =0
		senWordLen = 0
		sentList=data['sentences'][data['qid2cid'][idx]]
		for j in range(len(sentList)):
			# sentence tokenize rules
			senTokenLen = len(sentList[j])
			senWordLen += senTokenLen
			
			# sent bound rules
			if(j!=len(sentList)-1 and str(sentList[j][-1])!= "." and str(sentList[j][-1])!= "?" and str(sentList[j][-1])!= "!" ):
				continue
				if(len(sentList[j])>=2 and (str(sentList[j][-1])!= "\"" and str(sentList[j][-2])!= ".")):
					continue

			senEndWordIndex = senBeginWordIndex+senWordLen
			senIdxList.append([senBeginWordIndex, senEndWordIndex])
			senBeginWordIndex = senEndWordIndex
			senWordLen =0

		time2=time.time()
		# MAKE ANSWER SENTENCE INDEX LIST
		ansSenIdxList = []
		for i in range(len(ans_tokens)):
			# answer token is given as [begin index, end index]
			ans_begin_idx, ans_end_idx = ans_tokens[i]
			ans_end_idx_bound = ans_end_idx+1
			
			# parameters for answer through multiple sentences
			answerOverlapMaxLen = 0
			newAnsIdx = (0,0)
			newAnsIdxFlag = 0

			for j in range(len(senIdxList)):
				senBeginWordIndex, senEndWordIndex = senIdxList[j]
				if(senBeginWordIndex==senEndWordIndex):
					ansBESameCount += 1
				if(senBeginWordIndex <= ans_begin_idx and senEndWordIndex >= ans_end_idx_bound):
					ansSenIdxList.append(j)
					break
				# ANSWER THROUGH MULTIPLE SENTENCES
				else:
					
					answerOverlapLen = 0
					new_begin_idx = 0
					new_end_idx = 0
					if((senBeginWordIndex <= ans_begin_idx and senEndWordIndex > ans_begin_idx)):
						answerOverlapLen = senEndWordIndex-ans_begin_idx
						new_begin_idx = ans_begin_idx
						new_end_idx = senEndWordIndex-1
					if((senBeginWordIndex < ans_end_idx_bound and senEndWordIndex >= ans_end_idx_bound)):
						answerOverlapLen = ans_end_idx_bound-senBeginWordIndex
						new_begin_idx = senBeginWordIndex
						new_end_idx = ans_end_idx_bound-1
					if(answerOverlapLen > 0):
						# find max overlapping sentence
						if(len(ansSenIdxList)==i+1 and answerOverlapMaxLen < answerOverlapLen):
							ansSenIdxList[-1]=j
							answerOverlapMaxLen = answerOverlapLen
							newAnsIdx = (new_begin_idx,new_end_idx)
							newAnsIdxFlag = 1
						else:
							if(len(ansSenIdxList)<i+1):
								ansSenIdxList.append(j)
								newAnsIdx = (new_begin_idx,new_end_idx)
								newAnsIdxFlag = 1

			# answer index changed
			if(newAnsIdxFlag ==1 ):
					
				# sent bound error dict
				answer = ' '.join(document[ans_begin_idx:ans_end_idx_bound])
				answerSentenceList = []
				for senBeginWordIndex, senEndWordIndex in senIdxList:
					if((senBeginWordIndex <= ans_begin_idx and senEndWordIndex > ans_begin_idx) or (senBeginWordIndex < ans_end_idx_bound and senEndWordIndex >= ans_end_idx_bound)):
						answerSentenceList.append(' '.join(document[senBeginWordIndex:senEndWordIndex]))
				senBoundErrorDict[answer] = answerSentenceList

				if(ans_tokens[i][0] > newAnsIdx[0] or ans_tokens[i][1] < newAnsIdx[1]):
					print(ans_tokens[i])
					print(newAnsIdx)
					print("new answer boundary out of default answer boundary")
					exit()
				ans_tokens[i] = newAnsIdx
				ansOverlapCount += 1

			wholeAnswerCount += 1

		time3=time.time()
		# no answer sentence with answer
		if(len(ansSenIdxList)==0 and len(ans_tokens)!=0):
			emptySenError +=1

		# sentence token len sum not match with whole context length
		if(len(document)!=senIdxList[-1][1]):
			tokenCountError +=1

		# check ans count equals ans sen count
		if(len(ans_tokens)!=len(ansSenIdxList)):
			ansSenError += 1

		# check answer boundary fits to sentence boundary
		for begin_idx, end_idx in ans_tokens:
			exitFlag = 0
			for sen_begin_idx, sen_end_bound in senIdxList:
				if(sen_begin_idx <= begin_idx and sen_end_bound-1 >= end_idx):
					exitFlag = 1
			if(exitFlag != 1):
				print(begin_idx)
				print(end_idx)
				print(senIdxList)
				print("answer boundary out of sentence boundary")
				exit()

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
	print("whole answer count")
	print(wholeAnswerCount)
	print("answer overlapping count")
	print(ansOverlapCount)
	print("empty_sen_count")
	print(emptySenError)
	print("token count error")
	print(tokenCountError)
	print("ans sen error")
	print(ansSenError)
	print("ans be same error")
	print(ansBESameCount)

	senBoundError.write("num of cases:"+str(len(senBoundErrorDict)))
	senBoundError.write("\n")
	for key, value in senBoundErrorDict.items():
		senBoundError.write(key)
		senBoundError.write("\n")
		senBoundError.write(' // '.join(value))
		senBoundError.write("\n\n")


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
	args.out_dir, 'resultDir/%s-processed-%s.txt' % (args.split, args.tokenizer)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
	for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
		f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
