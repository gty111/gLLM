import random


class AllocatorID():
    def __init__(self, minnum=1, maxnum=1000):
        self.free_ids = list(range(minnum, maxnum+1))
        self.used_ids = []

    def allocate(self, id:int=None):
        if id is None:
            assert len(self.free_ids) != 0
            id = random.sample(self.free_ids, 1)[0]
            self.used_ids.append(id)
            self.free_ids.remove(id)
        else:
            if not id in self.used_ids:
                self.used_ids.append(id)
                self.free_ids.remove(id)
        return id

    def free(self, id: int):
        assert id in self.used_ids
        self.free_ids.append(id)
        self.used_ids.remove(id)

    def is_empty(self):
        return len(self.used_ids) == 0

    def get_num_used_ids(self):
        return len(self.used_ids)
    
    def get_num_free_ids(self):
        return len(self.free_ids)
    
    def get_num_ids(self):
        return len(self.free_ids) + len(self.used_ids)
