import torch
from bert_ect import EmotionClassifier
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer


class Gini:
    def __init__(self, tokenizer, device, model, checkpoint_path=None):
        self.device = device
        self.tokenizer = tokenizer
        self.generic_emotions_list = ['anger', 'revulsion', 'joy', 'passion', 'sadness', 'surprise', 'neutral']

        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path))
        self.model = model.to(device)
        self.model.eval()

    def label_to_emotion(self, label):
        return self.generic_emotions_list[label]

    def emotion(self, sentence):
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            return_tensors='pt',
            truncation=True
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # logit = outputs.logit
        probabilities = torch.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        predicted_emotion = self.label_to_emotion(predicted_label)

        return predicted_emotion, probabilities[0][predicted_label].item()


if __name__ == '__main__':
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = '../checkpoints/Aug_RoBerta_FT.pth'
    model = EmotionClassifier(7, is_roberta=True, mlp_enable=False)
    gini = Gini(roberta_tokenizer, device, model, model_path)
    while True:
        sentence = input("Tell me something:\n")
        feeling, confidence = gini.emotion(sentence)
        print(f"I feel you are in : {feeling}. Im sure with {confidence * 100}%")

