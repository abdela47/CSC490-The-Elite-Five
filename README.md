# CSC490-The-Elite-Five
## Project Name: Human Emotion Detection
## Collaborators: Ahmed Abdelaziz, Muhammad Haris Idrees, Hashem Al-Hussieny, Sahil Malek, Bonan Lin 

### The Problem
The problem we are solving is human facial emotion detection and identification. Given an image of a human face, we would like to classify the image with a human emotion. There are many real-world applications for this task. For example, emotion detection can be used in psychology to assist in understanding and aiding patients. It can also assist in any criminal interrogation and lie detection procedures. Finally, human emotion detection can enhance many social media applications, such as Snapchat’s face filters.

### Dataset
The dataset is Emotion Detection (2020), provided by Kaggle. This dataset contains more than 35,000 human facial images of various emotion. The images themselves are 48x48 pixel grayscale images. There are 7 classes of emotions - happiness, neutral, sadness, anger, surprise, disgust, fear. 

### Implementation

### Results

### Individual Contributions
#### Ahmed Abdelaziz's Contributions:
- Initial set up of repository and Google Collab
- Worked on preprocessing and found improvements
- Prefetching the data in training to speed up preprocessing
- Implementing the Initial model and applying Hashem's recommendation of normalization for preprocessing.
- After the initial model I looked into ideas of improvements for the model
- Recommended the use of the Adam optimizer and applied it to the first model.
- Assisted with recommendations and ideas for future model iterations. (support role)

#### Muhammad Haris Idrees
- Assisted with Model choice exploration. Deciding which model was appropriated for our problem.
- Involved with Model creation using multiple libraries, such as Sequential, Pytorch and even writing models using the CNN classes.
- Contributed to the training of the model such as overfitting models to small datasets to test usability.
- Testing Accuracy of Model with tuned parameters, and exploring optimization strategies.

#### Hashem Al-Hussieny’s Contributions:
- Worked on preprocessing
- Contributed to the initial implementation of the model
- Helped accelerate preprocessing in the training phase through the prefetching of data.
- Data cleaning/Split the data in a consistent manner in order for the model to train and test off of.
- Ensured data put into the model was consistent, through the addition of a normalization layer, optimizing the data, etc.

#### Bonan Lin’s Contribution:
- Tune the hyperparameter to increase the accuracy
- Do research to find the fittable model to our project
- Analyze the result to check if the result is overfitting or underfitting
- Summary each one’s poster and combine the final poster.

#### Sahil Malek’s Contribution:
- Model Training and Model Creation
- Hyperparameter Tuning such as filter sizes, epochs etc.
- Optimising model accuracy and training speed using batch normalization layers, varied convolutional layer filters and speeding up colab’s slow google drive reads.
- Exploring Models to find best fit for our problem, such as EfficientNet and AlexNet
- Testing Accuracy of Model


### Running The Model
- Please use Google Colab and Google Drive
- Upload a zip of the dataset to your Google Drive account. Paste the path to the zip file into two locations marked with "TODO: Replace with path to dataset zip in your drive". This is to ensure that Google Colab can unzip the file into its file system. This is done to overcome the large overhead of directly reading files from Google Drive


### References
- Hadeerismail. (2023, September 29). Emotion recognition. Kaggle. https://www.kaggle.com/code/hadeerismail/emotion-recognition
- Mrgrhn. (2021, January 21). Alexnet with tensorflow. Medium. https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8
- Hargurjeet. (2021, May 27). 7 best techniques to improve the accuracy of CNN w/O overfitting. Medium. https://medium.com/mlearning-ai/7-best-techniques-to-improve-the-accuracy-of-cnn-w-o-overfitting-6db06467182f
- Complete guide to the adam optimization algorithm. Built In. (n.d.). https://builtin.com/machine-learning/adam-optimization 
