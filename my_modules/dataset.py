import torch


class MyDataset(torch.utils.data.Dataset):
    """
        My Dataset with some class funcs
    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]  # the index th sequence in input_list
        input_ids = input_ids[:self.max_len]  # truncate the sequence to max_len
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)