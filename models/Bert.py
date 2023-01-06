from transformers import AutoTokenizer, BertForSequenceClassification

def tokenizer():
    return AutoTokenizer.from_pretrained('bert-base-uncased')
def transformer():
    return  BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=2)