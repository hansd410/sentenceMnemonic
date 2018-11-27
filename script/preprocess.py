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

	# count and error variable
	ansOverlapCount = 0
	ansBESameCount = 0
	wholeAnswerCount = 0
	sentCountDict = {}
	sentCount = open(args.logDir+"/sentCount_"+args.split+".txt",'w')
	parenErrorCount = 0
	quoErrorCount = 0

	emptySenError = 0
	tokenCountError = 0 
	ansSenError = 0
	sentBoundErrorDict = {}
	sentBoundError = open(args.logDir+"/sentBoundError_"+args.split+".txt",'w')

	# process
	for idx in range(len(data['qids'])):
		# read data
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
		
		ans_idx_tokens = []
		ans_text = []
		if len(data['answers']) > 0:
			for ans in data['answers'][idx]:
				found = find_answer(offsets,
									ans['answer_start'],
									ans['answer_start'] + len(ans['text']))
				if found:
					ans_idx_tokens.append(found)
					ans_text.append(ans['text'])

		# sentence index list generation
		sentIdxList = []
		sentBeginWordIndex =0
		sentWordLen = 0
		sentList=data['sentences'][data['qid2cid'][idx]]
		parenFlag = 0
		quoFlag = 0

		for j in range(len(sentList)):
			continueFlag =0
			sentTokenLen = len(sentList[j])
			sentWordLen += sentTokenLen
			
			# sent bound rules
			# rule 1. do not split without . ? !
			if(args.senSplitRules>0):
				if(j<len(sentList)-1 and
					str(sentList[j][-1])!= "." and str(sentList[j][-1])!= "?" and str(sentList[j][-1])!= "!" ):
					continueFlag=1
					if(len(sentList[j])>=2 and
						str(sentList[j][-2])!= "." and (str(sentList[j][-1])!= '"' )):
						continueFlag=1
			# rule 2. do not split . and "
			if(args.senSplitRules>1):
				if(j<len(sentList)-1 and
					(str(sentList[j][-1])== "." or str(sentList[j][-1])== "?" or str(sentList[j][-1])== "!") and
					str(sentList[j+1][0])=='"'):
					continueFlag=1
			# rule 3. if answer start with . , move one word
			if(args.senSplitRules>2):
				remove_idx_list = []
				for i in range(len(ans_idx_tokens)):
					ans_begin_idx, ans_end_idx = ans_idx_tokens[i]
					while(ans_begin_idx<len(document)-1 and (document[ans_begin_idx]=="." or document[ans_begin_idx]=="?" or document[ans_begin_idx]=="!")):
						ans_begin_idx += 1
					if(ans_begin_idx==len(document)-1):
						remove_idx_list.append(i)
					else:
						ans_idx_tokens[i]=ans_begin_idx,ans_end_idx
				remove_idx_list.sort(reverse=True)
				for i in range(len(remove_idx_list)):
					ans_idx_tokens.remove(ans_idx_tokens[remove_idx_list[i]])

			# rule 4. do not split Mr.
			if(args.senSplitRules>3):
				filterList = ["Mr","Mrs","Ms","etc","Op","St","Rs","Sch","PT","Ss","Msgr","al","Estep","ca","a.k.a","Fr","PA","Ft"]
				if(j<len(sentList)-1 and
					str(sentList[j][-1])== "." and
					str(sentList[j][-2]) in filterList ):
					continueFlag=1

			# rule 5. do not split paren
			if(args.senSplitRules>4):
				for wordIdx, word in enumerate(sentList[j]):
					if("(" in str(word) or "[" in str(word) or "<" in str(word)):
						parenFlag += str(word).count("(")
						parenFlag += str(word).count("[")
						parenFlag += str(word).count("<")
					if(")" in str(word) or "]" in str(word) or ">" in str(word)):
						parenFlag -= str(word).count(")")
						parenFlag -= str(word).count("]")
						parenFlag -= str(word).count(">")

					if('"' in str(word) and
						not( len(str(word))==1 and wordIdx>0 and str(sentList[j][wordIdx-1])[-1].isnumeric())  and
						not( len(str(word))!=1 and str(sentList[j][wordIdx])[str(sentList[j][wordIdx]).find('"')-1].isnumeric()) ):
						if(quoFlag ==0):
							quoFlag = 1
						else:
							quoFlag = 0

				if(parenFlag !=0 or quoFlag == 1):
					if(j==len(sentList)-1):
						if(parenFlag !=0):
							parenErrorCount += 1
						if(quoFlag == 1):
							quoErrorCount += 1
					else:
						continueFlag=1

			if(continueFlag == 1):
				continue
			else:
				sentEndWordIndex = sentBeginWordIndex+sentWordLen
				sentIdxList.append([sentBeginWordIndex, sentEndWordIndex])
				sentBeginWordIndex = sentEndWordIndex
				sentWordLen =0

		# count sentence num for debug
		if(len(sentIdxList) in sentCountDict.keys()):
			sentCountDict[len(sentIdxList)] += 1
		else:
			sentCountDict[len(sentIdxList)]=0

		# make answer sentence index list
		wholeAnswerCount += len(ans_idx_tokens)
		ansSenIdxList = []
		for i in range(len(ans_idx_tokens)):
			# answer token is given as [begin index, end index]
			ans_begin_idx, ans_end_idx = ans_idx_tokens[i]
			ans_end_idx_bound = ans_end_idx+1
			
			# parameters for answer through multiple sentences
			answerOverlapMaxLen = 0
			newAnsIdx = (0,0)
			newAnsIdxFlag = 0

			for j in range(len(sentIdxList)):
				sentBeginWordIndex, sentEndWordIndex = sentIdxList[j]
				if(sentBeginWordIndex==sentEndWordIndex):
					ansBESameCount += 1
				if(sentBeginWordIndex <= ans_begin_idx and sentEndWordIndex >= ans_end_idx_bound):
					ansSenIdxList.append(j)
					break
				# answer through multiple sentences
				else:
					answerOverlapLen = 0
					new_begin_idx = 0
					new_end_idx = 0
					if((sentBeginWordIndex <= ans_begin_idx and sentEndWordIndex > ans_begin_idx)):
						answerOverlapLen = sentEndWordIndex-ans_begin_idx
						new_begin_idx = ans_begin_idx
						new_end_idx = sentEndWordIndex-1
					if((sentBeginWordIndex < ans_end_idx_bound and sentEndWordIndex >= ans_end_idx_bound)):
						answerOverlapLen = ans_end_idx_bound-sentBeginWordIndex
						new_begin_idx = sentBeginWordIndex
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

			# change answer index
			if(newAnsIdxFlag ==1 ):
				ansOverlapCount += 1
				answer = ' '.join(document[ans_begin_idx:ans_end_idx_bound])
				answerSentenceList = []
				for sentBeginWordIndex, sentEndWordIndex in sentIdxList:
					if((sentBeginWordIndex <= ans_begin_idx and sentEndWordIndex > ans_begin_idx) or (sentBeginWordIndex < ans_end_idx_bound and sentEndWordIndex >= ans_end_idx_bound)):
						answerSentenceList.append(' '.join(document[sentBeginWordIndex:sentEndWordIndex]))
				sentBoundErrorDict[answer] = (answerSentenceList, question)
				ans_idx_tokens[i] = newAnsIdx

		# check answer index changing
		for begin_idx, end_idx in ans_idx_tokens:
			exitFlag = 0
			for sent_begin_idx, sent_end_bound in sentIdxList:
				if(sent_begin_idx <= begin_idx and sent_end_bound-1 >= end_idx):
					exitFlag = 1
			if(exitFlag != 1):
				print(len(document))
				print(document[begin_idx:end_idx+1])
				print(document[sentIdxList[-1][1]:])
				print(sentList)
				print(begin_idx)
				print(end_idx)
				print(sentIdxList)
				print("answer boundary out of sentence boundary")
				exit()

		# count errors
		# CHECK1 no answer sentence with answer
		if(len(ansSenIdxList)==0 and len(ans_idx_tokens)!=0):
			emptySenError +=1

		# CHECK2 sentence token len sum not match with whole context length
		if(len(document)!=sentIdxList[-1][1]):
			tokenCountError +=1

		# CHECK3 check ans count equals ans sen count
		if(len(ans_idx_tokens)!=len(ansSenIdxList)):
			ansSenError += 1

		# WRITE TO FILE
		yield {
			'id': data['qids'][idx],
			'question': question,
			'question_char': question_char,
			'document': document,
			'document_char': document_char,
			'offsets': offsets,
			'answers': ans_idx_tokens,
			'qlemma': qlemma,
			'qpos': qpos,
			'qner': qner,
			'clemma': clemma,
			'cpos': cpos,
			'cner': cner,
			'document_sentence': sentIdxList,
			'sentence_answers': ansSenIdxList,
		}

	# print count and error
	print("whole answer count")
	print(wholeAnswerCount)
	print("answer overlapping count")
	print(ansOverlapCount)
	print("ans be same error")
	print(ansBESameCount)

	sentCount.write("answer overlapping count : "+str(ansOverlapCount))
	sentCount.write("\n")
	sentCount.write("paren error count : "+str(parenErrorCount))
	sentCount.write("\n")
	sentCount.write("quo error count : "+str(quoErrorCount))
	sentCount.write("\n")
	for key,value in sorted(sentCountDict.items()):
		sentCount.write(str(key)+"\t"+str(value))
		sentCount.write("\n")

	print("empty_sen_count")
	print(emptySenError)
	print("token count error")
	print(tokenCountError)
	print("ans sen error")
	print(ansSenError)

	sentBoundError.write("num of cases:"+str(len(sentBoundErrorDict)))
	sentBoundError.write("\n")
	for key, value in sentBoundErrorDict.items():
		sentBoundError.write(' '.join(value[1]))
		sentBoundError.write("\n")
		sentBoundError.write(key)
		sentBoundError.write("\n")
		sentBoundError.write(' // '.join(value[0]))
		sentBoundError.write("\n\n")


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--log_dir', type=str, default='logFile', help='Path to log file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--tokenizer', type=str, default='spacy')
parser.add_argument('--senSplitRules', type=int, default=0)
args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
	args.out_dir, 'resultDir/%s-processed-%s-rule%s.txt' % (args.split, args.tokenizer, args.senSplitRules)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
	for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
		f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))
