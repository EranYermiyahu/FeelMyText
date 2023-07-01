from transformers import BertModel
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, GPT2ForSequenceClassification
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.1, is_roberta=True, feature_extracting=False, mlp_enable=False, n_layers=1, n_units=50):
        super(EmotionClassifier, self).__init__()

        if mlp_enable:
            ptm_output_dim = 28
            layers = []
            in_features = ptm_output_dim
            for i in range(n_layers):
                out_features = n_units
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_features = out_features
            layers.append(nn.Linear(in_features, num_classes))
            self.mlp = nn.Sequential(*layers)
        else:
            ptm_output_dim = num_classes

        self.mlp_enabled = mlp_enable

        # define pre-trained model
        if is_roberta:
            self.model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=ptm_output_dim,
                output_attentions=False,
                output_hidden_states=False)
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=ptm_output_dim)

        if feature_extracting:
            for name, param in self.model.named_parameters():
                if name.startswith('classifier'):  # Unfreeze the last layer
                    param.requires_grad = True
                else:  # Freeze all other parameters
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if self.mlp_enabled is False:
            return logits
        bert_dropped = logits
        output = self.mlp(bert_dropped)
        return output
    
        # out1 = self.fc1(bert_dropped)
        # pooled_out1 = self.drop(out1)
        # output = self.out(pooled_out1)
        # return output


# class EmotionClassifier(nn.Module):
#     def __init__(self, num_classes=7, dropout=0.1, is_roberta=True, feature_extracting=False, mlp_enable=False):
#         super(EmotionClassifier, self).__init__()
#         # self.bert = BertModel.from_pretrained('bert-base-uncased')
#         if mlp_enable:
#             ptm_output_dim = 28
#             self.drop = nn.Dropout(p=dropout)
#             self.fc1 = nn.Linear(28, 28)
#             self.out = nn.Linear(28, num_classes)
#         else:
#             ptm_output_dim = num_classes

    #     self.mlp_enabled = mlp_enable

    #     # define pre-trained model
    #     if is_roberta:
    #         self.model = RobertaForSequenceClassification.from_pretrained(
    #             'roberta-base',
    #             num_labels=ptm_output_dim,
    #             output_attentions=False,
    #             output_hidden_states=False)
    #     else:
    #         self.model = BertForSequenceClassification.from_pretrained(
    #             'bert-base-uncased',
    #             num_labels=ptm_output_dim)

    #     if feature_extracting:
    #         for name, param in self.model.named_parameters():
    #             if name.startswith('classifier'):  # Unfreeze the last layer
    #                 param.requires_grad = True
    #             else:  # Freeze all other parameters
    #                 param.requires_grad = False

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     logits = outputs.logits
    #     if self.mlp_enabled is False:
    #         return logits
    #     bert_dropped = self.drop(logits)
    #     out1 = self.fc1(bert_dropped)
    #     pooled_out1 = self.drop(out1)
    #     output = self.out(pooled_out1)
    #     return output