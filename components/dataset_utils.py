"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)

class LFCandidate:
    def __init__(self, s_expr, normed_expr, ex=None, f1=None, edist=None):
        self.s_expr = s_expr
        self.normed_expr = normed_expr
        self.ex = ex
        self.f1 = f1
        self.edist = edist

    def __str__(self):
        return '{}\n\t->{}\n'.format(self.s_expr, self.normed_expr)

    def __repr__(self):
        return self.__str__()
 