from transformers import BertModel
from transformers import BertForSequenceClassification
from transformers import AutoModel
import torch.nn as nn



class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = model = AutoModel.from_pretrained("nreimers/BERT-Tiny_L-2_H-128_A-2")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)