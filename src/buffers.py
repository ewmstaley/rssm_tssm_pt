'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np

class Buffer:
    def __init__(self, sizes, max_size=10000):
        self.collections = []
        for sz in sizes:
            self.collections.append(np.zeros((max_size, *sz)))
        self.max_size = max_size
        self.ptr = 0
        self.rolling_size = 0
        self.total_processed = 0

    def add_samples(self, samples):
        for sample in samples:
            if len(self.collections) == 1:
                sample = [sample]

            for i in range(len(sample)):
                self.collections[i][self.ptr] = sample[i]

            self.ptr += 1
            self.total_processed += 1
            self.rolling_size = min(self.rolling_size+1, self.max_size)
            if self.ptr >= self.max_size:
                self.ptr = 0

    def sample_buffers(self, amount=32):
        idx = np.random.choice(self.rolling_size, size=amount)
        rets = []
        for c in self.collections:
            rets.append(c[idx])
        if len(rets)==1:
            rets = rets[0]
        return rets


class SubTrajectoryBuffer(Buffer):
    # buffer that allows you to sample segments of trajectories.
    # assume that add_samples provides a complete trajectory.
    def __init__(self, sizes, max_size=10000):
        sizes = list(sizes)
        sizes.append((1,))
        self.start_indices = []
        super().__init__(sizes, max_size)

    def add_samples(self, samples):
        self.start_indices.append((self.ptr, self.total_processed))
        counts = [len(samples)-i-1 for i in range(len(samples))]
        if len(self.collections)==2:
            samples = [(samples[i], counts[i]) for i in range(len(samples))]
        else:
            samples = [(*samples[i], counts[i]) for i in range(len(samples))]
        super().add_samples(samples)

        # remove old data
        retained_start_indices = []
        for entry in self.start_indices:
            idx, t = entry
            if t >= self.total_processed - self.rolling_size:
                retained_start_indices.append((idx, t))
        self.start_indices = retained_start_indices

    def sample_buffers(self, amount=32):
        idx = np.random.choice(self.rolling_size, size=amount)
        rets = []
        for c in self.collections[:-1]:
            rets.append(c[idx])
        if len(rets)==1:
            rets = rets[0]
        return rets

    def sample_subtrajectories(self, subtraj_len=4, amount=32, start_from_zero=False):
        start_idxs = []

        if start_from_zero:
            options = np.array([x[0] for x in self.start_indices])
            if len(options) >= amount:
                options = options[np.random.permutation(len(options))]
                start_idxs = options[:amount]
            else:
                start_idxs = options[np.random.choice(len(options), size=amount)]
        else:
            while len(start_idxs) < amount:
                candidate = np.random.choice(self.rolling_size)
                if self.collections[-1][candidate] >= subtraj_len:
                    start_idxs.append(candidate)
            start_idxs = np.array(start_idxs)

        all_rets = []
        for i in range(subtraj_len):
            rets = []
            for c in self.collections[:-1]:
                rets.append(c[(start_idxs+i) % len(c)])
            if len(rets)==1:
                rets = rets[0]
            all_rets.append(rets)
        return all_rets




if __name__ == "__main__":

    stb = SubTrajectoryBuffer([(1,), (1,)], max_size=100)

    for i in range(9):
        states = np.arange(10) + 100*(i+1)
        actions = states + 10*(i+1)
        tups = [(states[k], actions[k]) for k in range(10)]
        stb.add_samples(tups)

    x = stb.sample_subtrajectories(4, 5)
    print(x[0][0], x[1][0], x[2][0], x[3][0])
    print(x[0][1], x[1][1], x[2][1], x[3][1])