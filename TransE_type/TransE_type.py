import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class TransE_type(nn.Module):
	def __init__(self, num_ent, num_rel, dim, p_norm, cuda=True):
		super(TransE_type, self).__init__()

		self.num_ent = num_ent
		self.num_rel = num_rel
		self.dim = dim
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.num_ent, self.dim)
		self.rel_embeddings = nn.Embedding(self.num_rel, self.dim)

		init_range = 6.0 / np.sqrt(self.dim)
		nn.init.uniform_(self.ent_embeddings.weight.data, -init_range, init_range)
		nn.init.uniform_(self.rel_embeddings.weight.data, -init_range, init_range)

	def forward(self, batch):
		h = self.ent_embeddings(batch[:, 0])
		t = self.ent_embeddings(batch[:, 1])
		r = self.rel_embeddings(batch[:, 2])

		h = F.normalize(h, self.p_norm, -1)
		t = F.normalize(t, self.p_norm, -1)
		r = F.normalize(r, self.p_norm, -1)

		res = h + r - t
		return -torch.norm(res, self.p_norm, -1)