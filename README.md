<h1 align="center">FeelMyText : Emotion Classifier From Text Using Data Augmentations</h1>
<h2 align="center">046211 - Deep Learning, The Faculty of Electrical and Computer Engineering, Technion - Israeli Institute of Technology
</h2> 

  <p align="center">
    Yossi Meshulam: <a href="https://www.linkedin.com/in/yossi-meshulam-90ab3517a/">LinkedIn</a> , <a href="https://github.com/yossimeshulam">GitHub</a>
  <br>
    Eran Yermiyahu: <a href="http://www.linkedin.com/in/eran-yermiyahu/">LinkedIn</a> , <a href="https://github.com/EranYermiyahu">GitHub</a>
  </p>

## Project Explanation Video (Hebrew)
https://www.youtube.com/watch?v=ePDFk_Pc8A4

## How to use FeelMyText?

First, download locally the model to your computer from this OneDrive link:
https://technionmail-my.sharepoint.com/:u:/r/personal/eeran_campus_technion_ac_il/Documents/Technion/Courses/Semester%20H/046211%20-%20Deep%20Learning/Project/FeelMyText/Aug_RoBerta_FT_MLP.pth?csf=1&web=1&e=ldEhvm

Save the model in a new directory called checkpoints in the following path from the src directory: '../checkpoints/Aug_RoBerta_FT.pth'
Please execute the code in src/gini.py
This will launch an interface for interacting with the model (Inference only). **Insert a sentence and FeelMytext will try to guess your feelings!**

```
./src/gini.py
```
* The main code stored in the src directory, including python classes files we used in order to train models, pre-processes data, search hyperparameters with Optuna and implement methods. Running main.py will train the 16 models provided in the presentation with fixed hyperparameters if global var OPTUNA=False, and will search for hyperparameters over specific model Aug_RoBERTa_FT_MLP otherwise

Example video of the model at work:
https://github.com/EranYermiyahu/FeelMyText/assets/82023584/54fb96b4-a901-4dbb-a40a-66be20fa9728

## Background
Our primary research objective is to establish a reliable dataset that serves as a fertile ground for training pre-trained transformer encoders capable of effectively classifying emotions within raw textual content. To achieve a dataset of utmost quality, we have employed text augmentation algorithms and meticulously evaluated their impact on both the performance and robustness of the model. Additionally, we seek to investigate the efficacy of leveraging the expressive capabilities of supplementary Multi-Layer Perceptron (MLP) layers atop pre-trained models. We will explore various approaches to transfer learning during the training process and examine the contribution of hyperparameter tuning towards optimizing the model's performance.


## Dataset
For our research, we have chosen the GoEmotions dataset, which was developed by Google and made publicly available. This dataset serves as our foundation for uncovering the underlying connections that encapsulate emotions within a given text. It comprises a human-annotated collection of 70,000 Reddit comments sourced from popular English-language subreddits, with each comment labeled according to one of 28 distinct emotion categories.

!<img width="559" alt="image" src="https://github.com/EranYermiyahu/FeelMyText/assets/73947067/99be60c1-1870-4e39-869a-2123ca3fff06">

## Data Augmentations
Creating a reliable dataset is crucial for any data-driven project. One aspect of dataset reliability is ensuring an equal number of samples for each label. In our project, we recognized the importance of balanced data and took specific steps to achieve this goal.

Our first step was to address the issue of class imbalance by generalizing the existing 28 labels into 7 primary labels. This consolidation allowed us to simplify the dataset while still capturing the essence of the original labels. By reducing the number of labels, we aimed to achieve a more equal distribution of samples across the dataset.

However, the process of generalization alone did not completely balance the dataset. To further equalize the distribution, we employed the powerful ChatGPT API. Leveraging the capabilities of this language model, we generated new samples from the existing ones. This approach enabled us to augment the dataset with additional instances, thereby increasing the representation of underrepresented labels.

Through meticulous implementation of these techniques, we successfully created a final dataset that consists of 9396 samples for each of the 7 primary labels. This equalized distribution ensures that no particular label dominates the dataset, providing a solid foundation for our subsequent analyses and model training.

By striving for an equal number of samples per label, we have enhanced the reliability of our dataset. This balance will not only prevent biases during model training but also enable the model to learn effectively across all label categories. With a robust and unbiased dataset, we can confidently proceed with our project, knowing that our results will be representative and dependable.

![image](https://github.com/EranYermiyahu/FeelMyText/assets/73947067/b3ecf406-be9d-4070-b83b-2203446077fa)
![image](https://github.com/EranYermiyahu/FeelMyText/assets/73947067/2a2a183a-2bdf-4b86-b6cd-9966aecbd168)

## Model Architecture

In our project, we employed the RoBERTa pre-trained model as the backbone of our classification system. RoBERTa, short for "Robustly Optimized BERT approach," is a state-of-the-art language model that has been trained on a massive amount of diverse text data. By utilizing this powerful pre-trained model, we leveraged its ability to understand and extract meaningful representations from natural language.

To adapt RoBERTa for our specific task of multi-label classification, we incorporated a Multi-Layer Perceptron (MLP) network on top of the model's outputs. The MLP network acts as a classifier, taking the RoBERTa embeddings as input and transforming them into predictions for each of the 7 labels in our dataset.

At the final layers of our model, we employed a softmax activation function to obtain probabilities for each label. This activation function ensures that the predicted probabilities sum up to 1, allowing us to interpret the outputs as probabilities of each label being present in the given input text.

To determine the final classification, we selected the label with the highest probability as the predicted label for a particular input. By choosing the label with the highest probability, we aimed to provide a single, most likely prediction for each text sample.

This architecture combining RoBERTa with an MLP network allows us to leverage the power of the pre-trained language model while tailoring it to our specific classification task. By employing this approach, we aimed to achieve accurate and reliable predictions for the 7 labels in our dataset.

Throughout the project, we fine-tuned the RoBERTa model using the labeled data, adjusting the model's weights to optimize its performance for our specific classification task. By fine-tuning the model on our dataset, we aimed to enhance its ability to understand the nuances and patterns specific to our labeled data.

Overall, the combination of the RoBERTa pre-trained model and the MLP classifier enabled us to build a robust and effective classification system, capable of accurately predicting the most likely label for a given input text based on the probabilities obtained from the model's final layers.

<img width="663" alt="image" src="https://github.com/EranYermiyahu/FeelMyText/assets/73947067/5724357a-d338-4ca4-ac51-929f3e2b0331">

## Files
```
src/                     
  main.py       # Main function
  dataset.py    # Dataset class
  bert_ect.py   # Model Class
  Trainer.py    # Trainer Class
  Gini.py       # Gini Class
```




https://github.com/EranYermiyahu/FeelMyText/assets/82023584/3bf965ae-3120-45f2-a84d-ed024e642951


