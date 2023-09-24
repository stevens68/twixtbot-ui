
class SelectSet:

    def __init__(self):

        self.item_by_index = []
        self.index_by_item = dict()

    def clone(self):

        copy = SelectSet()
        copy.item_by_index = list(self.item_by_index)
        copy.index_by_item = dict(self.index_by_item)
        return copy

    def add(self, x):

        if x in self.index_by_item:
            raise ValueError("duplicate element")

        self.index_by_item[x] = len(self.item_by_index)
        self.item_by_index.append(x)

    def remove(self, x):

        if x in self.index_by_item:
            index = self.index_by_item[x]
            if index == len(self.item_by_index) - 1:
                self.item_by_index.pop()
            else:
                move = self.item_by_index.pop()
                self.item_by_index[index] = move
                self.index_by_item[move] = index
            del self.index_by_item[x]

    def __contains__(self, x):

        return x in self.index_by_item

    def __len__(self):

        return len(self.item_by_index)

    def __getitem__(self, i):

        return self.item_by_index[i]

    def pick(self, rng):

        return rng.choice(self.item_by_index)
