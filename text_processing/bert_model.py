from transformers import BertTokenizer, BertModel
import torch

class BertEmbedding:
    def __init__(self):
        # Initialize the BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_embedding(self, text):
        # Tokenize and encode the text for BERT
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            # Get the last hidden states from the BERT model
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
