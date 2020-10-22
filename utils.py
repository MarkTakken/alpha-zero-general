class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)