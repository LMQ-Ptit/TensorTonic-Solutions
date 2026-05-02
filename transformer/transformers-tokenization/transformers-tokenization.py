import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Xây dựng từ điển: Special tokens trước, sau đó là từ duy nhất đã lowercase và sorted.
        """
        # Hint 1: Dùng enumerate để gán ID 0-3 cho special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        self.vocab_size = len(special_tokens)

        # Hint 2: Thu thập từ duy nhất, dùng .lower() và .split()
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word_to_id:
                    unique_words.add(word)
        
        # Hint 3: sorted() các từ duy nhất để ID mang tính deterministic
        for word in sorted(list(unique_words)):
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        """
        Chuyển text thành list ID (phải lowercase và dùng UNK cho từ lạ).
        """
        # Requirement: lowercase và split by whitespace
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token] # ID là 1
        
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Chuyển list ID thành text (dùng <UNK> cho ID lạ).
        """
        unk_str = self.unk_token # "<UNK>"
        
        return " ".join([self.id_to_word.get(i, unk_str) for i in ids])