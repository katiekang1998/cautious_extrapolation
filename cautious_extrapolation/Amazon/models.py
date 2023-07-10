
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch



class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs

def initialize_bert_based_model(d_out):

    model = DistilBertClassifier.from_pretrained(
        "distilbert-base-uncased",
        num_labels=d_out)

    return model

def initialize_bert_transform():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform