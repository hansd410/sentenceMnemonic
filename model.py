#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Document Reader model"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

from torch.autograd import Variable
from config import override_model_args
from r_net import R_Net
from rnn_reader import RnnDocReader
from m_reader import MnemonicReader
from data import Dictionary

logger = logging.getLogger(__name__)


class DocReader(object):
	"""High level model that handles intializing the underlying network
	architecture, saving, updating examples, and predicting examples.
	"""

	# --------------------------------------------------------------------------
	# Initialization
	# --------------------------------------------------------------------------

	def __init__(self, args, word_dict, char_dict, feature_dict,
				 state_dict=None, normalize=True):
		# Book-keeping.
		self.args = args
		self.word_dict = word_dict
		self.char_dict = char_dict
		self.args.vocab_size = len(word_dict)
		self.args.char_size = len(char_dict)
		self.feature_dict = feature_dict
		self.args.num_features = len(feature_dict)
		self.updates = 0
		self.use_cuda = False
		self.parallel = False

		# Building network. If normalize if false, scores are not normalized
		# 0-1 per paragraph (no softmax).
		if args.model_type == 'rnn':
			self.network = RnnDocReader(args, normalize)
		elif args.model_type == 'r_net':
			self.network = R_Net(args, normalize)
		elif args.model_type == 'mnemonic':
			self.network = MnemonicReader(args, normalize)
		else:
			raise RuntimeError('Unsupported model: %s' % args.model_type)

		# Load saved state
		if state_dict:
			# Load buffer separately
			if 'fixed_embedding' in state_dict:
				fixed_embedding = state_dict.pop('fixed_embedding')
				self.network.load_state_dict(state_dict)
				self.network.register_buffer('fixed_embedding', fixed_embedding)
			else:
				self.network.load_state_dict(state_dict)

	def expand_dictionary(self, words):
		"""Add words to the DocReader dictionary if they do not exist. The
		underlying embedding matrix is also expanded (with random embeddings).

		Args:
			words: iterable of tokens to add to the dictionary.
		Output:
			added: set of tokens that were added.
		"""
		to_add = {self.word_dict.normalize(w) for w in words
				  if w not in self.word_dict}

		# Add words to dictionary and expand embedding layer
		if len(to_add) > 0:
			logger.info('Adding %d new words to dictionary...' % len(to_add))
			for w in to_add:
				self.word_dict.add(w)
			self.args.vocab_size = len(self.word_dict)
			logger.info('New vocab size: %d' % len(self.word_dict))

			old_embedding = self.network.embedding.weight.data
			self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
														self.args.embedding_dim,
														padding_idx=0)
			new_embedding = self.network.embedding.weight.data
			new_embedding[:old_embedding.size(0)] = old_embedding

		# Return added words
		return to_add


	def expand_char_dictionary(self, chars):
		"""Add chars to the DocReader dictionary if they do not exist. The
		underlying embedding matrix is also expanded (with random embeddings).

		Args:
			chars: iterable of tokens to add to the dictionary.
		Output:
			added: set of tokens that were added.
		"""
		to_add = {self.char_dict.normalize(w) for w in chars
				  if w not in self.char_dict}

		# Add chars to dictionary and expand embedding layer
		if len(to_add) > 0:
			logger.info('Adding %d new chars to dictionary...' % len(to_add))
			for w in to_add:
				self.char_dict.add(w)
			self.args.char_size = len(self.char_dict)
			logger.info('New char size: %d' % len(self.char_dict))

			old_char_embedding = self.network.char_embedding.weight.data
			self.network.char_embedding = torch.nn.Embedding(self.args.char_size,
														self.args.char_embedding_dim,
														padding_idx=0)
			new_char_embedding = self.network.char_embedding.weight.data
			new_char_embedding[:old_char_embedding.size(0)] = old_char_embedding

		# Return added chars
		return to_add

	def load_embeddings(self, words, embedding_file):
		"""Load pretrained embeddings for a given list of words, if they exist.

		Args:
			words: iterable of tokens. Only those that are indexed in the
			  dictionary are kept.
			embedding_file: path to text file of embeddings, space separated.
		"""
		words = {w for w in words if w in self.word_dict}
		logger.info('Loading pre-trained embeddings for %d words from %s' %
					(len(words), embedding_file))
		embedding = self.network.embedding.weight.data

		# When normalized, some words are duplicated. (Average the embeddings).
		vec_counts = {}
		with open(embedding_file) as f:
			for line in f:
				parsed = line.rstrip().split(' ')
				assert(len(parsed) == embedding.size(1) + 1)
				w = self.word_dict.normalize(parsed[0])
				if w in words:
					vec = torch.Tensor([float(i) for i in parsed[1:]])
					if w not in vec_counts:
						vec_counts[w] = 1
						embedding[self.word_dict[w]].copy_(vec)
					else:
						logging.warning(
							'WARN: Duplicate embedding found for %s' % w
						)
						vec_counts[w] = vec_counts[w] + 1
						embedding[self.word_dict[w]].add_(vec)

		for w, c in vec_counts.items():
			embedding[self.word_dict[w]].div_(c)

		logger.info('Loaded %d embeddings (%.2f%%)' %
					(len(vec_counts), 100 * len(vec_counts) / len(words)))

	def load_char_embeddings(self, chars, char_embedding_file):
		"""Load pretrained embeddings for a given list of chars, if they exist.

		Args:
			chars: iterable of tokens. Only those that are indexed in the
			  dictionary are kept.
			char_embedding_file: path to text file of embeddings, space separated.
		"""
		chars = {w for w in chars if w in self.char_dict}
		logger.info('Loading pre-trained embeddings for %d chars from %s' %
					(len(chars), char_embedding_file))
		char_embedding = self.network.char_embedding.weight.data

		# When normalized, some chars are duplicated. (Average the embeddings).
		vec_counts = {}
		with open(char_embedding_file) as f:
			for line in f:
				parsed = line.rstrip().split(' ')
				assert(len(parsed) == char_embedding.size(1) + 1)
				w = self.char_dict.normalize(parsed[0])
				if w in chars:
					vec = torch.Tensor([float(i) for i in parsed[1:]])
					if w not in vec_counts:
						vec_counts[w] = 1
						char_embedding[self.char_dict[w]].copy_(vec)
					else:
						logging.warning(
							'WARN: Duplicate char embedding found for %s' % w
						)
						vec_counts[w] = vec_counts[w] + 1
						char_embedding[self.char_dict[w]].add_(vec)

		for w, c in vec_counts.items():
			char_embedding[self.char_dict[w]].div_(c)

		logger.info('Loaded %d char embeddings (%.2f%%)' %
					(len(vec_counts), 100 * len(vec_counts) / len(chars)))

	def tune_embeddings(self, words):
		"""Unfix the embeddings of a list of words. This is only relevant if
		only some of the embeddings are being tuned (tune_partial = N).

		Shuffles the N specified words to the front of the dictionary, and saves
		the original vectors of the other N + 1:vocab words in a fixed buffer.

		Args:
			words: iterable of tokens contained in dictionary.
		"""
		words = {w for w in words if w in self.word_dict}

		if len(words) == 0:
			logger.warning('Tried to tune embeddings, but no words given!')
			return

		if len(words) == len(self.word_dict):
			logger.warning('Tuning ALL embeddings in dictionary')
			return

		# Shuffle words and vectors
		embedding = self.network.embedding.weight.data
		for idx, swap_word in enumerate(words, self.word_dict.START):
			# Get current word + embedding for this index
			curr_word = self.word_dict[idx]
			curr_emb = embedding[idx].clone()
			old_idx = self.word_dict[swap_word]

			# Swap embeddings + dictionary indices
			embedding[idx].copy_(embedding[old_idx])
			embedding[old_idx].copy_(curr_emb)
			self.word_dict[swap_word] = idx
			self.word_dict[idx] = swap_word
			self.word_dict[curr_word] = old_idx
			self.word_dict[old_idx] = curr_word

		# Save the original, fixed embeddings
		self.network.register_buffer(
			'fixed_embedding', embedding[idx + 1:].clone()
		)

	def init_optimizer(self, state_dict=None):
		"""Initialize an optimizer for the free parameters of the network.

		Args:
			state_dict: network parameters
		"""
		if self.args.fix_embeddings:
			for p in self.network.embedding.parameters():
				p.requires_grad = False
		parameters = [p for p in self.network.parameters() if p.requires_grad]
		if self.args.optimizer == 'sgd':
			self.optimizer = optim.SGD(parameters, lr=self.args.learning_rate,
									   momentum=self.args.momentum,
									   weight_decay=self.args.weight_decay)
		elif self.args.optimizer == 'adamax':
			self.optimizer = optim.Adamax(parameters,
										  weight_decay=self.args.weight_decay)
		elif self.args.optimizer == 'adadelta':
			self.optimizer = optim.Adadelta(parameters, lr=self.args.learning_rate,
											rho=self.args.rho, eps=self.args.eps,
											weight_decay=self.args.weight_decay)
		else:
			raise RuntimeError('Unsupported optimizer: %s' %
							   self.args.optimizer)

	def sent_divided_nll_loss(self, scores_s, scores_e, scores_sent, targets_s, targets_e, sent_idx_list, ans_sent_idx_list):
		loss_s = 0
		loss_e = 0
		loss_sent = 0
		for score_s, score_e, score_sent, target_s, target_e, sent_idx, ans_sent_idx in zip(scores_s, scores_e, scores_sent, targets_s, targets_e, sent_idx_list, ans_sent_idx_list):
#			print("nll loss log")
#			print(score_s.size())
#			print(ans_sent_idx)
#			print(target_s)
#			print(target_e)
#			print(sent_idx[ans_sent_idx][0])
#			print(sent_idx)

			loss_s -= score_s[ans_sent_idx, target_s-sent_idx[ans_sent_idx][0]]
			loss_e -= score_e[ans_sent_idx, target_e-sent_idx[ans_sent_idx][0]]
			loss_sent -= score_sent[ans_sent_idx]
		if(self.args.sentence_attention==True):
			if(self.args.sentence_only ==True):
				return loss_sent
			else:
				# loss weight applied
				w = self.args.sent_loss_weight
				return (1-w)*(loss_s + loss_e) + w*loss_sent
		else:
			return loss_s + loss_e

	# --------------------------------------------------------------------------
	# Learning
	# --------------------------------------------------------------------------

	def update(self, ex):
		"""Forward a batch of examples; step the optimizer to update weights."""
		if not self.optimizer:
			raise RuntimeError('No optimizer set.')

		# Train mode
		self.network.train()
		
		# Transfer to GPU
		#print(len(ex))
		if self.use_cuda:
			inputs = [e if e is None else Variable(e.cuda(non_blocking=True)) for e in ex[:-5]] + [ex[-5]] + [[idx[0] for idx in ex[-4]]]
			target_s = Variable(ex[-3].cuda(non_blocking=True))
			target_e = Variable(ex[-2].cuda(non_blocking=True))
		else:
			inputs = [e if e is None else Variable(e) for e in ex[:-5]] + [ex[-5]] + [[idx[0] for idx in ex[-4]]]
			target_s = Variable(ex[-3])
			target_e = Variable(ex[-2])

		# Run forward
		# score_s, score_e = self.network(*inputs)
		score_s, score_e, score_sent = self.network(*inputs)

		# Compute loss and accuracies
		#loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e) 
		loss = self.sent_divided_nll_loss(score_s, score_e, score_sent, target_s, target_e, inputs[-2], inputs[-1])

		# Clear gradients and run backward
		self.optimizer.zero_grad()
		loss.backward()

		# Clip gradients
		torch.nn.utils.clip_grad_norm(self.network.parameters(),
									  self.args.grad_clipping)

		# Update parameters
		self.optimizer.step()
		self.updates += 1

		# Reset any partially fixed parameters (e.g. rare words)
		self.reset_parameters()

		return loss.data[0], ex[0].size(0)

	def reset_parameters(self):
		"""Reset any partially fixed parameters to original states."""

		# Reset fixed embeddings to original value
		if self.args.tune_partial > 0:
			# Embeddings to fix are indexed after the special + N tuned words
			offset = self.args.tune_partial + self.word_dict.START
			if self.parallel:
				embedding = self.network.module.embedding.weight.data
				fixed_embedding = self.network.module.fixed_embedding
			else:
				embedding = self.network.embedding.weight.data
				fixed_embedding = self.network.fixed_embedding
			if offset < embedding.size(0):
				embedding[offset:] = fixed_embedding

	# --------------------------------------------------------------------------
	# Prediction
	# --------------------------------------------------------------------------

	def sent_merged_score(self,scores_s, scores_e, scores_sent, sent_idx_list, ans_sent_idx_list):
		merged_score_s = scores_s.new()
		merged_score_s.resize_(scores_s.size(0),scores_s.size(1)*scores_s.size(2))
		merged_score_s.fill_(0)

		merged_score_e = scores_e.new()
		merged_score_e.resize_(scores_e.size(0),scores_s.size(1)*scores_e.size(2))
		merged_score_e.fill_(0)

		one_score_sent = scores_sent.new()
		one_score_sent.resize_(scores_sent.size(1))


		for i, (score_s, score_e, score_sent, sent_idx, ans_sent_idx) in enumerate(zip(scores_s, scores_e, scores_sent, sent_idx_list, ans_sent_idx_list)):
			# without sentence attention, first answer sentence chosen for test
			one_score_sent.fill_(0)
			one_score_sent[ans_sent_idx[0]].fill_(1)

			if(self.args.sentence_attention==True):
				score_s = score_s*score_sent.unsqueeze(1)
				score_e = score_e*score_sent.unsqueeze(1)
			else:
				score_s = score_s*one_score_sent.unsqueeze(1)
				score_e = score_e*one_score_sent.unsqueeze(1)

			for j in range(len(sent_idx)):
				merged_score_s[i,sent_idx[j][0]:sent_idx[j][1]] = score_s[j,0:sent_idx[j][1]-sent_idx[j][0]]
				merged_score_e[i,sent_idx[j][0]:sent_idx[j][1]] = score_e[j,0:sent_idx[j][1]-sent_idx[j][0]]

		return merged_score_s,merged_score_e


	def predict(self, ex, candidates=None, top_n=1, async_pool=None):
		"""Forward a batch of examples only to get predictions.

		Args:
			ex: the batch
			candidates: batch * variable length list of string answer options.
			  The model will only consider exact spans contained in this list.
			top_n: Number of predictions to return per batch element.
			async_pool: If provided, non-gpu post-processing will be offloaded
			  to this CPU process pool.
		Output:
			pred_s: batch * top_n predicted start indices
			pred_e: batch * top_n predicted end indices
			pred_score: batch * top_n prediction scores

		If async_pool is given, these will be AsyncResult handles.
		"""
		# Eval mode
		self.network.eval()

		# Transfer to GPU
#		if self.use_cuda:
#			inputs = [e if e is None else
#					  Variable(e.cuda(non_blocking=True), volatile=True)
#					  for e in ex[:8]]
#		else:
#			inputs = [e if e is None else Variable(e, volatile=True)
#					  for e in ex[:8]]
		# ex : x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, sen_idx_list, ans_sen_idx, y_s, y_e, ids
		if self.use_cuda:
			inputs = [e if e is None else Variable(e.cuda(non_blocking=True),volatile=True) for e in ex[:-5]] + [ex[-5]] + [ex[-4]]
		else:
			inputs = [e if e is None else Variable(e,volatile=True) for e in ex[:-5]] + [ex[-5]] + [ex[-4]]

		# Run forward
		# batch * senNum * featureDim
		score_s, score_e, score_sent = self.network(*inputs)
		score_s, score_e = self.sent_merged_score(score_s, score_e, score_sent, inputs[-2], inputs[-1])
		del inputs

		# Decode predictions
		score_s = score_s.data.cpu()
		score_e = score_e.data.cpu()

		if candidates:
			args = (score_s, score_e, candidates, top_n, self.args.max_len)
			if async_pool:
				return async_pool.apply_async(self.decode_candidates, args), score_sent
			else:
				return self.decode_candidates(*args), score_sent
		else:
			args = (score_s, score_e, top_n, self.args.max_len)
			if async_pool:
				return async_pool.apply_async(self.decode, args), score_sent
			else:
				return self.decode(*args), score_sent

	@staticmethod
	def decode(score_s, score_e, top_n=1, max_len=None):
		"""Take argmax of constrained score_s * score_e.

		Args:
			score_s: independent start predictions
			score_e: independent end predictions
			top_n: number of top scored pairs to take
			max_len: max span length to consider
		"""
		pred_s = []
		pred_e = []
		pred_score = []
		max_len = max_len or score_s.size(1)
		for i in range(score_s.size(0)):
			# Outer product of scores to get full p_s * p_e matrix
			scores = torch.ger(score_s[i], score_e[i])

			# Zero out negative length and over-length span scores
			scores.triu_().tril_(max_len - 1)

			# Take argmax or top n
			scores = scores.numpy()
			scores_flat = scores.flatten()
			if top_n == 1:
				idx_sort = [np.argmax(scores_flat)]
			elif len(scores_flat) < top_n:
				idx_sort = np.argsort(-scores_flat)
			else:
				idx = np.argpartition(-scores_flat, top_n)[0:top_n]
				idx_sort = idx[np.argsort(-scores_flat[idx])]
			s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
			pred_s.append(s_idx)
			pred_e.append(e_idx)
			pred_score.append(scores_flat[idx_sort])
		del score_s, score_e
		return pred_s, pred_e, pred_score

	@staticmethod
	def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
		"""Take argmax of constrained score_s * score_e. Except only consider
		spans that are in the candidates list.
		"""
		pred_s = []
		pred_e = []
		pred_score = []
		for i in range(score_s.size(0)):
			# Extract original tokens stored with candidates
			tokens = candidates[i]['input']
			cands = candidates[i]['cands']

			if not cands:
				# try getting from globals? (multiprocessing in pipeline mode)
				from ..pipeline.wrmcqa import PROCESS_CANDS
				cands = PROCESS_CANDS
			if not cands:
				raise RuntimeError('No candidates given.')

			# Score all valid candidates found in text.
			# Brute force get all ngrams and compare against the candidate list.
			max_len = max_len or len(tokens)
			scores, s_idx, e_idx = [], [], []
			for s, e in tokens.ngrams(n=max_len, as_strings=False):
				span = tokens.slice(s, e).untokenize()
				if span in cands or span.lower() in cands:
					# Match! Record its score.
					scores.append(score_s[i][s] * score_e[i][e - 1])
					s_idx.append(s)
					e_idx.append(e - 1)

			if len(scores) == 0:
				# No candidates present
				pred_s.append([])
				pred_e.append([])
				pred_score.append([])
			else:
				# Rank found candidates
				scores = np.array(scores)
				s_idx = np.array(s_idx)
				e_idx = np.array(e_idx)

				idx_sort = np.argsort(-scores)[0:top_n]
				pred_s.append(s_idx[idx_sort])
				pred_e.append(e_idx[idx_sort])
				pred_score.append(scores[idx_sort])
		del score_s, score_e, candidates
		return pred_s, pred_e, pred_score

	# --------------------------------------------------------------------------
	# Saving and loading
	# --------------------------------------------------------------------------

	def save(self, filename):
		state_dict = copy.copy(self.network.state_dict())
		if 'fixed_embedding' in state_dict:
			state_dict.pop('fixed_embedding')
		params = {
			'state_dict': state_dict,
			'word_dict': self.word_dict,
			'char_dict': self.char_dict,
			'feature_dict': self.feature_dict,
			'args': self.args,
		}
		try:
			torch.save(params, filename)
		except BaseException:
			logger.warning('WARN: Saving failed... continuing anyway.')

	def checkpoint(self, filename, epoch):
		params = {
			'state_dict': self.network.state_dict(),
			'word_dict': self.word_dict,
			'char_dict': self.char_dict,
			'feature_dict': self.feature_dict,
			'args': self.args,
			'epoch': epoch,
			'optimizer': self.optimizer.state_dict(),
		}
		try:
			torch.save(params, filename)
		except BaseException:
			logger.warning('WARN: Saving failed... continuing anyway.')

	@staticmethod
	def load(filename, new_args=None, normalize=True):
		logger.info('Loading model %s' % filename)
		saved_params = torch.load(
			filename, map_location=lambda storage, loc: storage
		)
		word_dict = saved_params['word_dict']
		try:
			char_dict = saved_params['char_dict']
		except KeyError as e:
			char_dict = Dictionary()

		feature_dict = saved_params['feature_dict']
		state_dict = saved_params['state_dict']
		args = saved_params['args']
		if new_args:
			args = override_model_args(args, new_args)
		return DocReader(args, word_dict, char_dict, feature_dict, state_dict, normalize)

	@staticmethod
	def load_checkpoint(filename, normalize=True):
		logger.info('Loading model %s' % filename)
		saved_params = torch.load(
			filename, map_location=lambda storage, loc: storage
		)
		word_dict = saved_params['word_dict']
		char_dict = saved_params['char_dict']
		feature_dict = saved_params['feature_dict']
		state_dict = saved_params['state_dict']
		epoch = saved_params['epoch']
		optimizer = saved_params['optimizer']
		args = saved_params['args']
		model = DocReader(args, word_dict, char_dict, feature_dict, state_dict, normalize)
		model.init_optimizer(optimizer)
		return model, epoch

	# --------------------------------------------------------------------------
	# Runtime
	# --------------------------------------------------------------------------

	def cuda(self):
		self.use_cuda = True
		self.network = self.network.cuda()

	def cpu(self):
		self.use_cuda = False
		self.network = self.network.cpu()

	def parallelize(self):
		"""Use data parallel to copy the model across several gpus.
		This will take all gpus visible with CUDA_VISIBLE_DEVICES.
		"""
		self.parallel = True
		self.network = torch.nn.DataParallel(self.network)
