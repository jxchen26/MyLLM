import re

text = None
with open("the-verdict.txt") as file:
    text = file.read()

#split a , . and whitespaces
result = re.split(r'([,.;:?_!"()\']|--|\s)',text)

#remove whitespaces ( reduce memory but lose sensitivity to text with whitespaces - like python code!)
preprocessed = [token.strip() for token in result if token.strip()]
print(len(preprocessed))

"""
converting tokens -> token IDs

(this is an intemediate step before converting the 
tokens IDs into embeddings vectors)

vocab define how we map each unique word/characters to a unique integer
"""

UNK = "<|unk|>"
EOT = "<|endoftext|>"
allWords = sorted (set(preprocessed))
allWords.extend(["<|endoftext|>", "<|unk|>"])
print(len(allWords))# 1130 + 2

#create vocab
vocab = {token:integer for integer,token in enumerate(allWords)}


class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab 
        self.int_to_str = {i:s for s,i in vocab.items()} # creates inverse vocab

    def encode (self, text: str ) :
        preprocessed = re.split(r'([,.;:?_!"()\']|--|\s)',text)
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        return [ self.str_to_int[ token if token in self.str_to_int else UNK ] for token in preprocessed]

    def decode(self, ids: list ):
        text = " ".join([self.int_to_str[id] for id in ids ])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #remove space between a punctuation
        return text



simplenizer = SimpleTokenizerV1(vocab) # vocab from 'the-verdict'

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = simplenizer.encode(text)
print(ids)
print(simplenizer.decode(ids)) # looks good

#why we need large datasets, 'Hello' not found in the "the verdict"
#beside EOT, UNK, other sequences tokenizers might include are [padding], [end of squence], [beginning of sequence]
# EOT can also replace [padding]
# chatGPT dont use word as tokens, but rather sub words for tokens ( byte pair encoding )
try:
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace." 
    text = " <|endoftext|> ".join((text1, text2))

    print(f"encoding text: {text}")
    print( simplenizer.decode(simplenizer.encode(text)))

except Exception as E:
    print(f'something went wrong while decoding "{text}". ')

"""

Byte pair encoding (BPE) (implementation is quite complicated)
in short this algorithm, builds the vocab by iteratively merging frequent characters into subwords 
and frequent subwords into words
1: encode "a","b".."z"
2. encode "un", "de"
3. encode: "undo", "depend"...
"""

from importlib.metadata import version
import tiktoken 
print(f"tiktoken version: {version('tiktoken')}")

tokenizer = tiktoken.get_encoding("gpt2")
text = ( "Hello, do you like tea? <|endoftext|> In the sunlit terraces" 
        "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) 
print(integers)

strings = tokenizer.decode(integers) 
print(strings)

text2 = "akwirw ier"
integers2 = tokenizer.encode(text2,allowed_special={"<|endoftext|>"}) 
print("akwirw ier ids:",integers2)

strings2 = tokenizer.decode(integers2) 
print("decoded string",strings2)

"""
data sampling with sliding windows

"""

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode (raw_text)
print(len(enc_text)) # should be 5145