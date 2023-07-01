# Augment using GPT sentences
import openai
import csv

# Load your OpenAI API key
openai.api_key = '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'

# Function to rephrase a sentence while preserving the emotion
def rephrase_sentence(sentence, emotion):
    # Specify the prompt with the desired sentence and emotion
    prompt = f"I feel {emotion} about this sentence: {sentence}\n\nRephrase the sentence, be creative:"
    #prompt = f"Rephrase the sentence, be creative and write the sentence to reflect the emotion {emotion}:\n{sentence}"
    if emotion in ['revulsion', 'sadness']:
      num_rep = 3
    else:
      num_rep = 2


    # Generate rephrased sentences
    response = openai.Completion.create(
        engine="text-curie-001",
        prompt=prompt,
        max_tokens=70,  # Adjust the number of tokens as needed
        n=num_rep,  # Number of alternative completions to generate
        temperature=0.8,  # Controls the randomness of the output
        stop=None  # Stop generation at a specific token (optional)
    )

    # Extract and return the rephrased sentences
    rephrased_sentences = [choice['text'].strip() for choice in response['choices']]
    return rephrased_sentences

# Load the CSV file
csv_file_path = 'train_filtered_readt_to_gpt.csv'
output_file_path = 'rephrased_filtered_csv.csv'

with open(csv_file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists

    # Initialize the output CSV file
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Original Sentence', 'Emotion', 'Rephrased Sentence 1', 'Rephrased Sentence 2'])

        # Iterate over each row in the CSV file
        for row in reader:
            sentence = row[0]  # Assuming the sentence is in the first column
            emotion = row[1]  # Assuming the emotion is in the second column
            emotion_list = ['anger', 'revulsion', 'joy', 'passion', 'sadness', 'surprise', 'neutral']
            emotion = emotion_list[int(emotion)]

            # Rephrase the sentence while preserving the emotion
            rephrased_sentences = rephrase_sentence(sentence, emotion)

            # Write the data to the output CSV file
            writer.writerow([sentence, emotion] + rephrased_sentences)

print("Rephrased sentences saved to:", output_file_path)