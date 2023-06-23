from transformers import BertModel
from transformers import BertForSequenceClassification
from transformers import AutoModel
import torch.nn as nn



# class EmotionClassifier(nn.Module):
#     def __init__(self, n_classes, dropout=0.3):
#         super(EmotionClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         # self.bert = model = AutoModel.from_pretrained("nreimers/BERT-Tiny_L-2_H-128_A-2")
#         self.drop = nn.Dropout(p=dropout)
#         self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs[0]
#         pooled_output = outputs[1]
#         output = self.drop(pooled_output)
#         return self.out(output)
    

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT parameters
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.classifier(pooled_output)