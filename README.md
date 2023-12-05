# CSC490-The-Elite-Five
## Project Name: Human Emotion Detection
## Collaborators: Ahmed Abdelaziz, Muhammad Haris Idrees, Hashem Al-Hussieny, Sahil Malek, Bonan Lin 

### The Problem
The problem we are solving is human facial emotion detection and identification. Given an image of a human face, we would like to classify the image with a human emotion. There are many real-world applications for this task. For example, emotion detection can be used in psychology to assist in understanding and aiding patients. It can also assist in any criminal interrogation and lie detection procedures. Finally, human emotion detection can enhance many social media applications, such as Snapchat’s face filters.

### Dataset
The dataset is Emotion Detection (2020), provided by Kaggle. This dataset contains more than 35,000 human facial images of various emotion. The images themselves are 48x48 pixel grayscale images. There are 7 classes of emotions - happiness, neutral, sadness, anger, surprise, disgust, fear. 

### Implementation
#### Model Implementation Decisions:
  We needed to be able to make decisions and classify images based on the features in the image of a facial expression. We explored many different model types such as AlexNet, but ultimately decided to use a Convolutional Neural Network. Our decision of using a CNN, stemmed from our need to extract features from images and recognize patterns in the emotion expression images. CNN's convolutional layers are able to extract and keep key information from the image while reducing its dimensionality. 
  Subsequent to our model choice, we began to construct the architecture of our model. We began with designing our model on paper and computing the dimensions of our image through each convolutional layer. We initially started with three convolutional layers with max pooling operations and a dropout layer. The model initially was quite simple, where we ran into issues in training. The training either would be really slow or produce poor accuracy. 
  By trying many different model architects we decided to increase the number of Convolutional layers, because we suspected that our model may be too simple. We added more dropout layers to improve training, and batch normalization layers. By doing this, we had sped up our training and achieved an accuracy of 57%. 

#### Innovations, novel approaches or exciting aspects:
Although we did look at other model implementations, we decided to create our own model from scratch, to make it unique to the model we explored. We continued to research aspects our model may require to speed up training, but we designed the model architecture on our own through a process of constant trial and error. Although we achieve an accuracy slightly lower than accuracies online, we took an innovative approach in solving the problem, that keeps our model unique. 


#### Incompleteness:
  In the preprocessing stage, since our main dataset was in black and white and the images were of people in different head positions the variety in location of facial features and the shadows being casted in the image made it very difficult to improve the models accuracy through preprocessing. We attempted various things, like enhancing contrast, image cropping, noise reduction however they did not significantly improve our accuracy. When we attempted to add these additional preprocessing steps to improve the accuracy of our model the time it took for it to train increased significantly. This is because the majority of our work was done on Google Colab and the limitations that we experienced and had to overcome were due to its inconsistency and inefficiency which hindered our ability to improve the accuracy of our project. 
  Improvements to our project would be to utilize a coloured dataset to begin with as the initial variation in color would have assisted in the preprocessing of the image and finding facial features. The main dataset we used was in black and white, this lack of variety in colors made it hard to isolate facial features due to it getting washed out or creating artifacts. This was one of the more significant problems that we faced in this project. We believed that the lack of color would assist us in improving training time however we know realize that the decrease in training time may have created a glass ceiling that we recognised too late into our project.


### Evaluation Results 
The loss in our project was reduced well. When the loss is close to 0, we stop it early to save the running time.The training accuracy is pretty good in the following diagram. The final accuracy of our training data is approaching 93%. The training accuracy increases dramatically at the beginning and slower at the end of several iterations. The validation accuracy is increasing as well! It increases from 10% to 57% in the end.  The running time is also control in an acceptable range, average running time takes around 70 seconds.


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
