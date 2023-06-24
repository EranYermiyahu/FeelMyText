from transformers import BertModel
from transformers import BertForSequenceClassification
from transformers import AutoModel
import torch.nn as nn



class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, dropout, feature_extracting=True):
        super(EmotionClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=28,
            output_attentions=False,
            output_hidden_states=False
        )
        if feature_extracting:
            for param in self.bert.parameters():
                param.requires_grad = False  # Freeze BERT parameters
        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(28, 28)
        self.out = nn.Linear(28, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        bert_dropped = self.drop(logits)
        out1 = self.fc1(bert_dropped)
        pooled_out1 = self.drop(out1)
        output = self.out(pooled_out1)
        return output
    

# class EmotionClassifier(nn.Module):
#     def __init__(self, n_classes):
#         super(EmotionClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         for param in self.bert.parameters():
#             param.requires_grad = False  # Freeze BERT parameters
#         self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]
#         return self.classifier(pooled_output)