import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TrainData, ValidData
from TransE_type import TransE_type
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import random


if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='Arguments')
	parser.add_argument('--neg', type = int, default = 25)
	parser.add_argument('--dim', type = float, default = 128)
	parser.add_argument('--epochs', type = int, default = 1000)
	parser.add_argument('--valid_epochs', type = int, default = 50)
	parser.add_argument('--lamb', type = float, default = 0.01)
	parser.add_argument('--data', type = str, default = None)
	parser.add_argument('--lr', type = float, default = 2.0)
	parser.add_argument('--margin', type = float, default = 1.0)
	parser.add_argument('--test', type = int, default = 0)
	args = parser.parse_args()

	random.seed(1234)
	trainData = TrainData(
		path_data = "../data/"+ args.data +"/",
		num_neg = args.neg
	)

	model = TransE_type(
		num_ent = trainData.num_ent,
		num_rel = trainData.num_rel,
		dim = args.dim,
		p_norm = 2
	)

	model.cuda()

	trainDataLoader = DataLoader(trainData, batch_size = len(trainData) // 100, shuffle = True)

	validData = ValidData(
		path_data = "../data/"+ args.data +"/",
		test = args.test
	)

	validDataLoader = DataLoader(validData, batch_size = 1, shuffle = False)

	optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
	loss = nn.MarginRankingLoss(margin = args.margin)

	train = tqdm(range(args.epochs))
	for epoch in train:
		total_loss = 0.0
		for i, (pos, neg) in enumerate(trainDataLoader):
			batch = [pos]
			for item in neg:
				batch.append(item)
			batch = torch.cat(batch, 0)
			score = model(batch.cuda())
			score_pos = score[:len(pos)]
			score_pos = torch.repeat_interleave(score_pos, trainData.num_neg)
			score_neg = score[len(pos):]
			output = loss(score_pos, score_neg, torch.ones(len(score_pos)).cuda())

			ent_embeddings = F.normalize(model.ent_embeddings.weight.data, 2, -1)
			# Similarity
			center = []
			for key in trainData.type2id:
				center.append(torch.mean(ent_embeddings[trainData.type2id[key]], axis=0))

			c = torch.zeros(model.num_ent, model.dim).cuda()
			for idx, key in enumerate(trainData.type2id):
				c[trainData.type2id[key]] = center[idx]
			loss_sim = torch.mean(torch.norm(c - ent_embeddings, model.p_norm, -1))

			loss_final = output + args.lamb * loss_sim 
			loss_final.backward()
			optimizer.step()
			total_loss += loss_final.item()

		train.set_description("Epoch {} | {:.4f}".format(epoch, total_loss))

		if epoch % args.valid_epochs == 0:
			model.eval()
			with torch.no_grad():
				mr = 0.0
				mrr = 0.0
				hit10 = 0
				hit3 = 0
				hit1 = 0
				for triplet in tqdm(validDataLoader):
					h, t, r = triplet[0]
					
					h = h.item()
					t = t.item()
					r = r.item()

					type_h = validData.id2type[h]
					type_t = validData.id2type[t]

					batch_head = triplet[0].repeat(validData.num_ent, 1)
					batch_head[:, 0] = torch.arange(validData.num_ent)
					batch_tail = triplet[0].repeat(validData.num_ent, 1)
					batch_tail[:, 1] = torch.arange(validData.num_ent)

					score_head = model(batch_head.cuda()).cpu().numpy()
					
					score_target_head = score_head[h]
					score_head[validData.filter_head[(t, r)]] = score_target_head - 1
					score_head = score_head[validData.type2id[type_h]]
					rank = len(np.where(score_head > score_target_head)[0]) + 1
					mr += rank
					mrr += 1 / rank
					if rank < 2:
						hit1 += 1
					if rank < 4:
						hit3 += 1
					if rank < 11:
						hit10 += 1

					score_tail = model(batch_tail.cuda()).cpu().numpy()
					score_target_tail = score_tail[t]
					score_tail[validData.filter_tail[(h, r)]] = score_target_tail - 1
					score_tail = score_tail[validData.type2id[type_t]]
					rank = len(np.where(score_tail > score_target_tail)[0]) + 1
					mr += rank
					mrr += 1 / rank
					if rank < 2:
						hit1 += 1
					if rank < 4:
						hit3 += 1
					if rank < 11:
						hit10 += 1
				print("[Epoch: {}]".format(epoch))
				print("MR:", mr / (2 * len(validData)))
				print("MRR:", mrr / (2 * len(validData)))
				print("Hit@10:", hit10 / (2 * len(validData)))
				print("Hit@3:", hit3 / (2 * len(validData)))
				print("Hit@1:", hit1 / (2 * len(validData)))
			model.train()