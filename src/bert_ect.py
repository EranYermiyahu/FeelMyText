from transformers import BertModel
from transformers import XLNetForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import AutoModel
import torch.nn as nn



class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, dropout, feature_extracting=False, mlp_enable=False):
        super(EmotionClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        if mlp_enable:
            roberta_output_dim = 28
            self.drop = nn.Dropout(p=dropout)
            self.fc1 = nn.Linear(28, 28)
            self.out = nn.Linear(28, num_classes)
        else:
            roberta_output_dim = num_classes
        self.mlp_enabled = mlp_enable
        self.bert = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=roberta_output_dim,
            output_attentions=False,
            output_hidden_states=False)

        if feature_extracting:
            for param in self.bert.parameters():
                param.requires_grad = False  # Freeze BERT parameters

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if self.mlp_enabled is False:
            return logits
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