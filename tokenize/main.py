import re

def tokenize_text(text):
    # Split text into words using whitespace as delimiter
    tokens = text.split()
    return tokens

def main():
    # Sample text for tokenization
    text = "Hello, how are you doing today?"

    # Tokenize the text
    tokens = tokenize_text(text)
    
    print("Original text:", text)
    print("Tokens:", tokens)

if __name__ == "__main__":
    main()
