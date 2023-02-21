import numpy as np
from torch.utils.data import Dataset

class TrainData(Dataset):
	def __init__(self, path_data, num_neg):
		self.path_data = path_data
		self.num_ent = self.num_instance("entity2id.txt")
		self.num_rel = self.num_instance("relation2id.txt")
		self.num_train, self.train = self.read_triplet("train2id.txt")
		self.num_type = self.num_instance("type2id.txt")
		self.type2id, self.id2type = self.read_type("entity2typeid.txt")

		self.filter_head, self.filter_tail = self.build_filter_dict()
		self.num_neg = num_neg

	def __len__(self):
		return self.num_train

	def __getitem__(self, idx):
		pos = self.train[idx]
		neg = self.negative_sampling(pos)
		return pos, neg

	def num_instance(self, target):
		with open(self.path_data + target, 'r') as f:
			res = int(f.readline().strip())
		return res

	def read_triplet(self, target):
		list_tri = []
		with open(self.path_data + target, 'r') as f:
			num_tri = int(f.readline().strip())
			for line in f.readlines():
				h, t, r = line.strip().split(' ')
				list_tri.append([int(h), int(t), int(r)])
		list_tri = np.array(list_tri)
		return num_tri, list_tri

	def read_type(self, target):
		type2id = dict()
		id2type = []
		with open(self.path_data + target, 'r') as f:
			for line in f.readlines():
				idx, typeid = line.strip().split(' ')
				idx = int(idx)
				typeid = int(typeid)

				if typeid not in type2id:
					type2id[typeid] = []
				type2id[typeid].append(idx)
				id2type.append(typeid)
		return type2id, id2type

	def build_filter_dict(self):
		filter_head = dict()
		filter_tail = dict()
		for h, t, r in self.train:
			if (h, r) not in filter_tail:
				filter_tail[(h, r)] = []
			filter_tail[(h, r)].append(t)

			if (t, r) not in filter_head:
				filter_head[(t, r)] = []
			filter_head[(t, r)].append(h)

		for item in filter_head:
			filter_head[item] = np.array(filter_head[item])
		for item in filter_tail:
			filter_tail[item] = np.array(filter_tail[item])

		return filter_head, filter_tail

	def negative_sampling(self, triplet):
		h, t, r = triplet
		type_h = self.id2type[h]
		type_t = self.id2type[t]
		count_h = len(self.type2id[type_h])
		count_t = len(self.type2id[type_t])

		candidate_head = np.setdiff1d(self.type2id[type_h], self.filter_head[(t, r)])
		candidate_tail = np.setdiff1d(self.type2id[type_t], self.filter_tail[(h, r)])
		corrupt = np.random.rand(self.num_neg)

		res = np.tile(triplet, [self.num_neg, 1])
		corrupt_head = np.where(corrupt < count_h / (count_h + count_t))[0]
		corrupt_tail = np.where(corrupt >= count_h / (count_h + count_t))[0]

		res[corrupt_head, 0] = np.random.choice(candidate_head, len(corrupt_head))
		res[corrupt_tail, 1] = np.random.choice(candidate_tail, len(corrupt_tail))
		return res

class ValidData(Dataset):
	def __init__(self, path_data, test):
		self.path_data = path_data
		self.test = test
		self.num_ent = self.num_instance("entity2id.txt")
		self.num_rel = self.num_instance("relation2id.txt")
		self.num_train, self.train = self.read_triplet("train2id.txt")
		self.num_valid, self.valid = self.read_triplet("valid2id.txt")
		self.num_test, self.test = self.read_triplet("test2id.txt")
		self.num_type = self.num_instance("type2id.txt")
		self.type2id, self.id2type = self.read_type("entity2typeid.txt")

		self.filter_head, self.filter_tail = self.build_filter_dict()

	def __len__(self):
		if self.test:
			return self.num_test
		else:
			return self.num_valid

	def __getitem__(self, idx):
		if self.test:
			return self.test[idx]
		else:
			return self.valid[idx]

	def num_instance(self, target):
		with open(self.path_data + target, 'r') as f:
			res = int(f.readline().strip())
		return res

	def read_triplet(self, target):
		list_tri = []
		with open(self.path_data + target, 'r') as f:
			num_tri = int(f.readline().strip())
			for line in f.readlines():
				h, t, r = line.strip().split(' ')
				list_tri.append([int(h), int(t), int(r)])
		list_tri = np.array(list_tri)
		return num_tri, list_tri

	def read_type(self, target):
		type2id = dict()
		id2type = []
		with open(self.path_data + target, 'r') as f:
			for line in f.readlines():
				idx, typeid = line.strip().split(' ')
				idx = int(idx)
				typeid = int(typeid)

				if typeid not in type2id:
					type2id[typeid] = []
				type2id[typeid].append(idx)
				id2type.append(typeid)
		return type2id, id2type

	def build_filter_dict(self):
		filter_head = dict()
		filter_tail = dict()
		for data in [self.train, self.valid, self.test]:
			for h, t, r in data:
				if (h, r) not in filter_tail:
					filter_tail[(h, r)] = []
				filter_tail[(h, r)].append(t)

				if (t, r) not in filter_head:
					filter_head[(t, r)] = []
				filter_head[(t, r)].append(h)

		for item in filter_head:
			filter_head[item] = np.array(filter_head[item])
		for item in filter_tail:
			filter_tail[item] = np.array(filter_tail[item])

		return filter_head, filter_tail