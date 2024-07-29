# Byte Pair Encoding

class Tokenizer:
    def __init__(self, training_src="data.txt", desiredVocabSize=356):
        f = open(training_src, "r", encoding="utf-8")
        text = f.read()

        self.original_tokens = list(text.encode("utf-8"))
        print("Number of Tokens Loaded: ", len(self.original_tokens))
        self.vocab_size = desiredVocabSize
        self.n_merges = desiredVocabSize - 256
        f.close()

        self.bytes_dict = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.train()


    def stats(self, tokens):
        pair_counts = {}
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def merge(self, tokens, pair, idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i<len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1]==pair[1]:
                new_tokens.append(idx)
                i+=2
            else:
                new_tokens.append(tokens[i])
                i+=1
        return new_tokens

    def train(self):

        tokens = list(self.original_tokens)
        for i in range(self.n_merges):
            print(f"Completed {i+1} Merges!")
            pair_counts = self.stats(tokens)
            pair = max((v, k) for k, v in pair_counts.items())[1]
            idx = i + 256

            tokens = self.merge(tokens, pair, idx)
            self.merges[pair] = idx

        
        for (p1, p2), idx in self.merges.items():
            self.bytes_dict[idx] = self.bytes_dict[p1] + self.bytes_dict[p2]

        print(f"Did {self.n_merges} Merges!")
        print(f"Number of Tokens after Training: {len(tokens)}")

        print(f"Compression Rate: {len(self.original_tokens)/len(tokens):.2f}X")


    def decode(self, tokens):
        text = b"".join([self.bytes_dict[i] for i in tokens])
        return text.decode("utf-8", errors="replace")


    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            pair_counts = self.stats(tokens)
            pair = min(pair_counts, key=lambda p: self.merges.get(p) if p in self.merges else float("inf"))
            if pair in self.merges:
                tokens = self.merge(tokens, pair, self.merges[pair])
            else:
                break
        return tokens
