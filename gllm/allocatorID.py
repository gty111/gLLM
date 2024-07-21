import random


class AllocatorID():
    def __init__(self, minnum=1, maxnum=1000):
        self.max_id = maxnum
        self.ids = list(range(minnum, maxnum+1))
        self.used_ids = []

    def allocate(self):
        assert len(self.ids) != 0
        id = random.sample(self.ids, 1)[0]
        self.used_ids.append(id)
        self.ids.remove(id)
        return id

    def free(self, id: int):
        assert id in self.used_ids
        self.ids.append(id)
        self.used_ids.remove(id)

    def is_empty(self):
        return len(self.used_ids) == 0

    def get_num_used_ids(self):
        return len(self.used_ids)
