import os 
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class SimpleTokenizer:
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        
        self.special_tokens = [
            "<|endoftext|>", 
            "<|im_start|>", 
            "<|im_end|>", 
            "<|padding|>", 
            "<|unknown|>"
        ]
        
        self.tokenizer.add_special_tokens(self.special_tokens)
        
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|padding|>"
        self.unk_token = "<|unknown|>"

    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

def train_tokenizer(input_file, save_path, vocab_size=49152):
    tokenizer = Tokenizer(BPE(unk_token="<|unknown|>"))
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [
        "<|endoftext|>", 
        "<|im_start|>", 
        "<|im_end|>", 
        "<|padding|>", 
        "<|unknown|>"
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel().alphabet(),
        show_progress=True
    )

    print(f"Training tokenizer... (Vocab: {vocab_size})...")
    tokenizer.train([input_file], trainer=trainer)
    
    tokenizer.save(save_path)
    print(f"Tokenizer saved : {save_path}")
