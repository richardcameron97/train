# train
import copy
import unittest

import numpy as np

from transformers.data.data_collator import default_data_collator
from transformers.testing_utils import require_accelerate, require_torch
from transformers.trainer_utils import RemoveColumnsCollator, find_executable_batch_size
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import IterableDataset

    from transformers.modeling_outputs import SequenceClassifierOutput
    from transformers.tokenization_utils_base import BatchEncoding
    from transformers.trainer_pt_utils import (
        DistributedLengthGroupedSampler,
        DistributedSamplerWithLoop,
        DistributedTensorGatherer,
        IterableDatasetShard,
        LabelSmoother,
        LengthGroupedSampler,
        SequentialDistributedSampler,
        ShardSampler,
        get_parameter_names,
        numpy_pad_and_concatenate,
        torch_pad_and_concatenate,
    )

    class TstLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
            self.bias = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            h = self.ln1(nn.functional.relu(self.linear1(x)))
            h = nn.functional.relu(self.linear2(x))
            return self.ln2(x + h + self.bias)

    class RandomIterableDataset(IterableDataset):
        # For testing, an iterable dataset of random length
        def __init__(self, p_stop=0.01, max_length=1000):
            self.p_stop = p_stop
            self.max_length = max_length
            self.generator = torch.Generator()

        def __iter__(self):
            count = 0
            stop = False
            while not stop and count < self.max_length:
                yield count
                count += 1
                number = torch.rand(1, generator=self.generator).item()
                stop = number < self.p_stop

