from collections import deque


class AllocatorID():
    def __init__(self, minnum=1, maxnum=1000):
        self.size = maxnum - minnum + 1
        self.free_ids = deque(range(minnum, maxnum+1))

    def allocate(self, id:int=None):
        if id is None:
            assert len(self.free_ids) != 0
            id = self.free_ids.popleft()
        else:
            if id in self.free_ids:
                self.free_ids.remove(id)
        return id

    def free(self, id: int):
        self.free_ids.appendleft(id)

    def is_empty(self):
        return len(self.free_ids) == self.size

    def get_num_used_ids(self):
        return self.size - len(self.free_ids)
    
    def get_num_free_ids(self):
        return len(self.free_ids)
