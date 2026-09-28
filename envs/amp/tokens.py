"""GFNx's AMP alphabet, preserving the pretrained policy's token IDs."""

AMINO_ACIDS = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]"]
PROTEINS_FULL_ALPHABET = AMINO_ACIDS + SPECIAL_TOKENS
