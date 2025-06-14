from collections import deque


class IDAllocator():
    def __init__(self, start_num=1, end_num=1000):
        self.size = end_num - start_num + 1
        self.free_ids = deque(range(start_num, end_num+1))

    def allocate(self, id:int=None):
        if id is None:
            assert len(self.free_ids) != 0
            id = self.free_ids.popleft()
        else:
            if id in self.free_ids:
                self.free_ids.remove(id)
        return id

    def free(self, id: int):
        # append instead of appendleft to improve prefix cache hit rate
        self.free_ids.append(id)

    def get_num_used_ids(self):
        return self.size - len(self.free_ids)
    
    def get_num_free_ids(self):
        return len(self.free_ids)
