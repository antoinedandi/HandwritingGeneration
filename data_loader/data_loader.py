from base import BaseDataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import plot_stroke, preprocess_sent, pad_collate


class HandWritingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.sentences_path = os.path.join(data_dir, 'sentences.txt')
        self.strokes_path = os.path.join(data_dir, 'strokes-py3.npy')

        self.sentences_file = open(self.sentences_path, encoding="utf8")

        self.sentences = [list(preprocess_sent(sent)) for sent in self.sentences_file.readlines()]
        self.strokes = np.load(self.strokes_path, encoding='latin1', allow_pickle=True)

        self.all_chars = self.find_all_chars()
        self.char2idx = {char: idx+1 for idx, char in enumerate(self.all_chars)}
        self.idx2char = {idx: char for (char, idx) in self.char2idx.items()}

    def find_all_chars(self):
        all_chars = set()
        for line in self.sentences:
            all_chars.update(line)
        sorted_chars = sorted(list(all_chars))  # important to keep the same char2idx across experience
        return sorted_chars

    def sentence2tensor(self, sentence):
        return torch.tensor(data=[self.char2idx[char] for char in sentence],
                            dtype=torch.long)

    def tensor2sentence(self, tensor):
        return ''.join([self.idx2char[idx] for idx in tensor.data.numpy()])

    def stroke2tensor(self, stroke):
        return torch.tensor(data=stroke,
                            dtype=torch.float)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        item = (self.sentences[idx], self.strokes[idx])
        item = (self.sentence2tensor(item[0]), self.stroke2tensor(item[1]))
        return item


class HandWritingDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, collate_fn=pad_collate):
        self.data_dir = data_dir
        self.dataset = HandWritingDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)


if __name__ == '__main__':
    data_directory = '../data'
    dataset = HandWritingDataset(data_directory)
    dataloader = HandWritingDataLoader(data_directory, batch_size=32, shuffle=False)

    # Test dataset
    print('Test dataset')
    print('size of the dataset: ', len(dataset))
    sent, stroke = dataset[0]
    print('sentence 1: ', dataset.tensor2sentence(sent))
    plot_stroke(stroke.numpy())

    # Test dataloader
    print('\nTest dataloader')
    batch = next(iter(dataloader))
    (sentences, sentences_mask, strokes, strokes_mask) = batch
    print('shape sentences:      ', sentences.shape)
    print('shape sentences_mask: ', sentences_mask.shape)
    print('shape strokes:        ', strokes.shape)
    print('shape strokes_mask:   ', strokes_mask.shape)

