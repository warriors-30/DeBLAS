import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info, Dataset
from multiprocessing import Array, Lock
import ctypes
import random

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        '''
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)
        '''
        iter_data = iter(self.data)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class LengthBucketDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length, split_ranges, initial_ratio):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.split_ranges = split_ranges.copy()

        # Shared tensor for ratio (works across worker processes)
        self._validate_ratio(initial_ratio)
        self.ratio_tensor = torch.tensor(initial_ratio, dtype=torch.float32).share_memory_()

    def _validate_ratio(self, ratio):
        assert len(ratio) == len(self.split_ranges), "Ratio length must match split_ranges"
        assert abs(sum(ratio) - 1.0) < 1e-6, "Ratio must sum to 1"

    def set_ratio(self, new_ratio):
        self._validate_ratio(new_ratio)
        self.ratio_tensor.copy_(torch.tensor(new_ratio, dtype=torch.float32))

    def __iter__(self):
        '''
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)
        '''
        iter_data = iter(self.data)

        buffers = [[] for _ in self.split_ranges]
        # for example: split_ranges = [(0,128),(128,256),(256,257)]
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            attention_mask = tokenized_example['attention_mask']
            length = attention_mask.sum().item()

            bucket_idx = None
            for idx, (start, end) in enumerate(self.split_ranges):
                if start <= length < end:
                    bucket_idx = idx
                    break
            if bucket_idx is None:
                continue

            buffers[bucket_idx].append(tokenized_example)
            current_ratio = self.ratio_tensor.tolist()  # Read shared value
            current_counts = [int(self.batch_size * r) for r in current_ratio]
            current_counts[-1] = self.batch_size - sum(current_counts[:-1])

            if all(len(buf) >= count for buf, count in zip(buffers, current_counts)):
                selected = []
                for i in range(len(buffers)):
                    count = current_counts[i]
                    selected.extend(buffers[i][:count])
                    buffers[i] = buffers[i][count:]
                yield self._format_batch(selected)

        # Yield remaining data (if any) with last ratio
        if all(len(buf) >= count for buf, count in zip(buffers, current_counts)):
            selected = []
            for i in range(len(buffers)):
                count = current_counts[i]
                selected.extend(buffers[i][:count])
                buffers[i] = buffers[i][count:]
            yield self._format_batch(selected)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SpecifiedLengthDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, specified_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.specified_length = specified_length

    def __iter__(self):
        iter_data = iter(self.data)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.specified_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            attention_mask = tokenized_example['attention_mask']
            length = attention_mask.sum().item()
            if length == self.specified_length:
                batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}