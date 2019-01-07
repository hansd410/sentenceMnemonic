#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the Mnemonic Reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class pwimNet(nn.Module):
	def __init__(self):
		super().__init__()
		def make_conv(n_in,n_out):
			conv = nn.Conv2d(n_in,n_out,3,padding=1)
			conv.bias.data.zero_()
			nn.init.xavier_normal_(conv.weight)
			return conv
		self.conv1 = make_conv(12,128)
		self.conv2 = make_conv(128,164)
		self.conv3 = make_conv(164,192)
		self.conv4 = make_conv(192,192)
		self.conv5 = make_conv(192,128)
		self.maxpool2 = nn.MaxPool2d(2,ceil_mode=True)
		self.dnn = nn.Linear(128,128)
		self.output = nn.Linear(128,1)
		self.input_len = 32

	# pad zeros to make pad*pad size
	def hard_pad2d(self,x,pad):
		def pad_side(idx):
			pad_len = max(pad-x.size(idx),0)
			return [0,pad_len]
		padding=pad_side(3)
		padding.extend(pad_side(2))
		x = F.pad(x,padding)
		return x[:,:,:pad,:pad]

	def forward(self,x):
		batchSize = x.size(0)
		x = x.permute(0,3,1,2,4).contiguous()
		x = x.view(-1,x.size(2),x.size(3),x.size(4))# (batch * sent num) * 12 * query len * context len
		x = self.hard_pad2d(x,self.input_len)
		x = self.maxpool2(F.relu(self.conv1(x)))
		x = self.maxpool2(F.relu(self.conv2(x)))
		x = self.maxpool2(F.relu(self.conv3(x)))
		x = self.maxpool2(F.relu(self.conv4(x)))
		x = self.maxpool2(F.relu(self.conv5(x)))
		x = F.relu(self.dnn(x.view(x.size(0),-1)))
		x = self.output(x).squeeze().contiguous().view(batchSize,-1)
		return x
	
class MnemonicReader(nn.Module):
	RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
	CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
	def __init__(self, args, normalize=True):
		super(MnemonicReader, self).__init__()
		# Store config
		self.args = args

		# Word embeddings (+1 for padding)
		self.embedding = nn.Embedding(args.vocab_size,
									  args.embedding_dim,
									  padding_idx=0)

		# Char embeddings (+1 for padding)
		self.char_embedding = nn.Embedding(args.char_size,
									  args.char_embedding_dim,
									  padding_idx=0)

		# Char rnn to generate char features
		self.char_rnn = layers.StackedBRNN(
			input_size=args.char_embedding_dim,
			hidden_size=args.char_hidden_size,
			num_layers=1,
			dropout_rate=args.dropout_rnn,
			dropout_output=args.dropout_rnn_output,
			concat_layers=False,
			rnn_type=self.RNN_TYPES[args.rnn_type],
			padding=args.rnn_padding,
		)

		doc_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.num_features

		# Encoder
		self.encoding_rnn = layers.StackedBRNN(
			input_size=doc_input_size,
			hidden_size=args.hidden_size,
			num_layers=1,
			dropout_rate=args.dropout_rnn,
			dropout_output=args.dropout_rnn_output,
			concat_layers=False,
			rnn_type=self.RNN_TYPES[args.rnn_type],
			padding=args.rnn_padding,
		)

		doc_hidden_size = 2 * args.hidden_size
		
		# Interactive aligning, self aligning and aggregating
		self.interactive_aligners = nn.ModuleList()
		self.interactive_SFUs = nn.ModuleList()
		self.self_aligners = nn.ModuleList()
		self.self_SFUs = nn.ModuleList()
		self.aggregate_rnns = nn.ModuleList()
		for i in range(args.align_hop):
			# interactive aligner
			self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
			self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
			# self aligner
			self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
			self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
			# aggregating
			self.aggregate_rnns.append(
				layers.StackedBRNN(
					input_size=doc_hidden_size,
					hidden_size=args.hidden_size,
					num_layers=1,
					dropout_rate=args.dropout_rnn,
					dropout_output=args.dropout_rnn_output,
					concat_layers=False,
					rnn_type=self.RNN_TYPES[args.rnn_type],
					padding=args.rnn_padding,
				)
			)

		# Memmory-based Answer Pointer
		self.mem_ans_ptr = layers.MemoryAnsPointer(
			x_size=2*args.hidden_size, 
			y_size=2*args.hidden_size, 
			hidden_size=args.hidden_size, 
			answer_hop=args.answer_hop,
			dropout_rate=args.dropout_rnn,
			normalize=normalize
		)

		# MLP for sentence projections for sentnce embeddings
		if(self.args.sentence_sewon):
			self.sent_w = nn.Linear(args.hidden_size * 2, 1, bias=False)
			self.sent_w2 = nn.Linear(args.hidden_size * 2, (args.hidden_size * 2)**2, bias=False)
			self.sent_w3 = nn.Linear(args.hidden_size * 2, 1, bias=False)
		else:
			if(self.args.sentence_cnn):
				self.pwim = pwimNet()
			else:
				self.sent_embedding = nn.Conv1d(args.hidden_size * 2, args.hidden_size * 2, 1, 1, 0)


	def forward(self, x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, sent_idx_list, ans_sent_idx_list):
		"""Inputs:
		x1 = document word indices			 [batch * len_d]
		x1_c = document char indices		   [batch * len_d]
		x1_f = document word features indices  [batch * len_d * nfeat]
		x1_mask = document padding mask		[batch * len_d]
		x2 = question word indices			 [batch * len_q]
		x2_c = document char indices		   [batch * len_d]
		x1_f = document word features indices  [batch * len_d * nfeat]
		x2_mask = question padding mask		[batch * len_q]

		sentence_index_list
		answer_sentence_index_list
		"""


		# Embed both document and question
		x1_emb = self.embedding(x1)
		x2_emb = self.embedding(x2)
		x1_c_emb = self.char_embedding(x1_c)
		x2_c_emb = self.char_embedding(x2_c)

		# Dropout on embeddings
		if self.args.dropout_emb > 0:
			x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
			x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
			x1_c_emb = F.dropout(x1_c_emb, p=self.args.dropout_emb, training=self.training)
			x2_c_emb = F.dropout(x2_c_emb, p=self.args.dropout_emb, training=self.training)

		# Generate char features
		x1_c_features = self.char_rnn(x1_c_emb, x1_mask)
		x2_c_features = self.char_rnn(x2_c_emb, x2_mask)

		# Combine input
		crnn_input = [x1_emb, x1_c_features]
		qrnn_input = [x2_emb, x2_c_features]
		# Add manual features
		if self.args.num_features > 0:
			crnn_input.append(x1_f)
			qrnn_input.append(x2_f)

		# Encode document with RNN
		c = self.encoding_rnn(torch.cat(crnn_input, 2), x1_mask)# batch * datalen * fitdim
		
		# Encode question with RNN
		q = self.encoding_rnn(torch.cat(qrnn_input, 2), x2_mask)# batch * datalen * fitdim

		if(self.args.encode_fix ==True):
			c=c.detach()
			q=q.detach()

		# Sangdo. sentence attention of Min
		if(self.args.sentence_sewon):
			sent_word_emb = c.new()
			sent_word_emb.resize_(c.size(0), max([len(sent) for sent in sent_idx_list]), 
								   max([max([idx[1]-idx[0] for idx in sent]) for sent in sent_idx_list]),
								   c.size(2)) # batch * max sent len * max word len * fit_dim
			sent_word_emb.fill_(0)

			for i, sent_idx in enumerate(sent_idx_list):
				for j, (begin, end) in enumerate(sent_idx):
					sent_word_emb[i, j, :end-begin, :] = c[i, begin:end, :]

			masked_q = q.masked_fill(x2_mask.unsqueeze(2).expand_as(q), 0)# batch_size x dataMaxLen x fitdim

			beta = F.softmax(self.sent_w(masked_q),dim=1) # batch_size x dataMaxLen x 1
			q_enc = torch.squeeze((beta * masked_q).sum(dim=1),1) # batch * fit_dim

			d_w = self.sent_w2(sent_word_emb).reshape(sent_word_emb.size(0),sent_word_emb.size(1),sent_word_emb.size(2),self.args.hidden_size * 2,self.args.hidden_size * 2) # batch * sent Num * sent Len * fitdim * fitdim
			#print(d_w.size())

			# reshape for matmul
			q_enc=q_enc.unsqueeze(1).unsqueeze(1).unsqueeze(-1).expand(q_enc.size(0),d_w.size(1),d_w.size(2),q_enc.size(1),1)
			temp = torch.squeeze(torch.matmul(d_w,q_enc),-1)
			h = temp.max(dim=2)[0]

			sent_score = torch.squeeze(self.sent_w3(h),-1) # batch size * max sentence count

			for i, sent_idx in enumerate(sent_idx_list):
				if len(sent_idx) != sent_score.size(1):
					sent_score[i, len(sent_idx):] = -1e10
		else:
			# sangdo. PWIM
			if(self.args.sentence_cnn):
				# MAKE SIM CUBE
				sent_word_emb = c.new()
				sent_word_emb.resize_(c.size(0), max([len(sent) for sent in sent_idx_list]), 
									   max([max([idx[1]-idx[0] for idx in sent]) for sent in sent_idx_list]),
									   c.size(2)) # batch * max sent len * max word len * fit_dim
				sent_word_emb.fill_(0)

				sent_word_mask = c.new()
				sent_word_mask.resize_(c.size(0), max([len(sent) for sent in sent_idx_list]), 
									   max([max([idx[1]-idx[0] for idx in sent]) for sent in sent_idx_list])) # batch * max sent len * max word len
				sent_word_mask.fill_(0)

				for i, sent_idx in enumerate(sent_idx_list):
					for j, (begin, end) in enumerate(sent_idx):
						sent_word_emb[i, j, :end-begin, :] = c[i, begin:end, :]
						sent_word_mask[i, j, :end-begin] = 1

				sim_cube = c.new()
				sim_cube.resize_(sent_word_emb.size(0),12,q.size(1),sent_word_emb.size(1),sent_word_emb.size(2)) # batch * 12* query word len * max sent len * max sent word len
				q_f = q[:,:,self.args.hidden_size:]
				q_b = q[:,:,:self.args.hidden_size]
				sent_word_emb_f = sent_word_emb[:,:,:,self.args.hidden_size:]
				sent_word_emb_b = sent_word_emb[:,:,:,:self.args.hidden_size]

				def compute_sim(query_prism,sent_word_prism):
					query_prism_len = query_prism.norm(dim=4)
					sent_word_prism_len = sent_word_prism.norm(dim=4)

					dot_prod = torch.matmul(query_prism.unsqueeze(4),sent_word_prism.unsqueeze(5))
					dot_prod = dot_prod.squeeze(4).squeeze(4)
					cos_dist = dot_prod / (query_prism_len*sent_word_prism_len+1E-8)
					l2_dist = (query_prism-sent_word_prism).norm(dim=4)
					return torch.stack([dot_prod,cos_dist,l2_dist],1)
				def compute_prism(query, sent_word): 
					query_prism = query.repeat(sent_word.size(1),sent_word.size(2),1,1,1)
					sent_word_prism = sent_word.repeat(query.size(1),1,1,1,1)
					query_prism = query_prism.permute(2,3,0,1,4).contiguous()
					sent_word_prism = sent_word_prism.permute(1,0,2,3,4).contiguous()
					return compute_sim(query_prism, sent_word_prism)

				sim_cube[:,0:3,:,:,:]=compute_prism(q,sent_word_emb)
				sim_cube[:,3:6,:,:,:]=compute_prism(q_f,sent_word_emb_f)
				sim_cube[:,6:9,:,:,:]=compute_prism(q_b,sent_word_emb_b)
				sim_cube[:,9:12,:,:,:]=compute_prism(q_f+q_b,sent_word_emb_f+sent_word_emb_b)

				# make pad cube
				query_mask = (1-x2_mask).type(c.type()).repeat(sim_cube.size(1),sim_cube.size(3),sim_cube.size(4),1,1) # x2_mask, padding =1 
				query_mask = query_mask.permute(3,0,4,1,2)
				sent_mask = sent_word_mask.repeat(sim_cube.size(1),sim_cube.size(2),1,1,1)
				sent_mask = sent_mask.permute(2,0,1,3,4)
				pad_cube = 1-query_mask*sent_mask # padding = 1

				truncate = self.pwim.input_len
				if truncate is not None:
					sim_cube = sim_cube[:,:,:truncate,:,:truncate].contiguous()
					pad_cube = pad_cube[:,:,:sim_cube.size(2),:,:sim_cube.size(4)].contiguous()

				# MAKE FOCUS CUBE
				neg_magic = -10000
				sim_cube = neg_magic*pad_cube+sim_cube
				mask = c.new()
				mask = mask.resize_(*pad_cube.size())
				mask[:,:,:,:,:] = 0.1

				# make mask
				def build_mask(index):
					# batch * query word len * max sent len * max sent word len
					max_mask = sim_cube[:,index].clone()
					for _ in range(min(sim_cube.size(2),sim_cube.size(4))):
						values,indices = torch.max(max_mask.view(sim_cube.size(0),sim_cube.size(3),-1),2)
						row_indices = indices/sim_cube.size(4) # query indices
						col_indices = indices%sim_cube.size(4) # sent indices
						#row_indices = row_indices.unsqueeze()
						#col_indices = col_indices.unsqueeze().unsqueeze()
						for i, (row_i,col_i,val) in enumerate(zip(row_indices,col_indices,values)):
							for j in range(max_mask.size(2)):
								if (val[j] < (neg_magic/2)):
									continue
								mask[i,:,row_i[j],j,col_i[j]]=1
								max_mask[i,row_i[j],j,:]=neg_magic
								max_mask[i,:,j,col_i[j]]=neg_magic
				
				build_mask(9) 
				build_mask(10)
				focus_cube = mask*sim_cube*(1-pad_cube)

				# APPLY PWIM
				sent_score = self.pwim(focus_cube)
				for i, sent_idx in enumerate(sent_idx_list):
					if len(sent_idx) != sent_score.size(1):
						sent_score[i, len(sent_idx):] = -1e10
			# Paul. sentence attention addition
			else:
				sent_word_emb = c.new()
				sent_word_emb.resize_(c.size(0), max([len(sent) for sent in sent_idx_list]), 
									   max([max([idx[1]-idx[0] for idx in sent]) for sent in sent_idx_list]),
									   c.size(2)) # batch * max sent len * max word len * fit_dim
				sent_word_emb.fill_(-1e10)


				for i, sent_idx in enumerate(sent_idx_list):
					for j, (begin, end) in enumerate(sent_idx):
						sent_word_emb[i, j, :end-begin, :] = c[i, begin:end, :]
				sent_word_emb = sent_word_emb.max(dim=2)[0]  # batch_size x max_sent_size x feat_dim

				sent_emb = self.sent_embedding(sent_word_emb.transpose(1, 2))  # batch_size x projected_feature_dim x max_sent_size

				q_proj = self.sent_embedding((q.masked_fill(x2_mask.unsqueeze(2).expand_as(q), -float('inf'))).max(dim=1)[0].unsqueeze(2))  # batch_size x project_feature_dim x 1
				sent_score = (sent_emb * q_proj).sum(dim=1)  # batch_size x max_sent_size

				for i, sent_idx in enumerate(sent_idx_list):
					if len(sent_idx) != sent_score.size(1):
						sent_score[i, len(sent_idx):] = -1e10
	
		if self.training:
			sent_score = F.log_softmax(sent_score, dim=1)
		else:
			sent_score = F.softmax(sent_score, dim=1)
		# Align and aggregate
		c_check = c
		for i in range(self.args.align_hop):
			q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
			c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
			c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
			c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
			c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)

		# Predict
		start_scores, end_scores = self.mem_ans_ptr.forward(c_check, q, x1_mask, x2_mask, sent_idx_list, ans_sent_idx_list)
		
		return start_scores, end_scores, sent_score
