from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
from torch.utils.data import Dataset, DataLoader


PAD = "[PAD]"
UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"


def train_tokenizer(texts, vocab_size=32000):
    tokenizer = Tokenizer(BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD, UNK, BOS, EOS],
    )

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def load_translation_data():
    dataset = load_dataset("opus_books", "de-en")
    train_data = dataset["train"].train_test_split(test_size=0.05, seed=42)

    train_set = train_data["train"]
    valid_set = train_data["test"]

    return train_set, valid_set


def get_text_iterator(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, max_len=128):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        self.pad_id = tgt_tokenizer.token_to_id(PAD)
        self.bos_id = tgt_tokenizer.token_to_id(BOS)
        self.eos_id = tgt_tokenizer.token_to_id(EOS)

    def encode(self, text, tokenizer):
        ids = tokenizer.encode(text).ids
        ids = ids[: self.max_len - 2]
        return [tokenizer.token_to_id(BOS)] + ids + [tokenizer.token_to_id(EOS)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]["translation"]

        src = self.encode(item["en"], self.src_tokenizer)
        tgt = self.encode(item["de"], self.tgt_tokenizer)

        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch, pad_id=0):
    src_batch, tgt_batch = zip(*batch)

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_id,
    )

    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_id,
    )

    return src_batch, tgt_batch


def build_dataloaders(batch_size=32, vocab_size=16000, max_len=128):
    train_set, valid_set = load_translation_data()

    src_tokenizer = train_tokenizer(
        get_text_iterator(train_set, "en"),
        vocab_size=vocab_size,
    )

    tgt_tokenizer = train_tokenizer(
        get_text_iterator(train_set, "de"),
        vocab_size=vocab_size,
    )

    train_dataset = TranslationDataset(
        train_set,
        src_tokenizer,
        tgt_tokenizer,
        max_len=max_len,
    )

    valid_dataset = TranslationDataset(
        valid_set,
        src_tokenizer,
        tgt_tokenizer,
        max_len=max_len,
    )

    pad_id = src_tokenizer.token_to_id(PAD)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id),
    )

    return train_loader, valid_loader, src_tokenizer, tgt_tokenizer
