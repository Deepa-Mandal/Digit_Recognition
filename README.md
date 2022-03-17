# Digit Recognition by using SVM

The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.

The https://www.kaggle.com/oddrationale/mnist-in-csv  dataset contains 60,000 training images of handwritten digits from zero to nine and 10,000 images for testing. So, the given dataset has 10 different classes. The handwritten digits images are represented as a 28×28 matrix.

Below are the steps to implement the handwritten digit recognition project:

## 1. Import the libraries and load the dataset
First, we imported all the libraries that we are going to need for training our model. We can easily import the dataset and start working with it. Then we install opendatasets to download the dataset from the link given to us.
## 2. Data Exploration and Visualization
The image training and testing data were downloaded as csv files. We can read the csv files and can show the columns of training and testing data. The dimension of the training data is (60000,28,28) and testing data is (10000,28,28). We can able to visualize the number of class and the counts in the datasets by using sns.countplot(). Then we examined and plotted few pixels 0, 3, 6 etc. The we created a heatmap which represented relationship of the values with the dataframe.
## 3. Data  Preprocessing
The image data cannot be fed directly into the model so we need to perform some operations and process the data to make it ready for our model creation. First we averaging the feature values and then separating the X_train and y_train variable by dropping the level column. As the pixel values vary in the range from 0-255, we divide the variable by 255 to normalize it. Then we split the training dataset into train test split as 20% and 80%.

## 4. Create the model
Now we will create our SVM model in Python project by using Gaussian Kernel and making predictions on training dataset. Then we create confusion matrix to show the result and print the classification report of the prediction by calculating precision, recall, f1-score and support. My model gives 96.2375% accuracy.
## 5. Grid Search: Hyperparameter Tuning
Now, we use 5-fold cross validation to tune the model and find the optimal values of C and gamma corresponding to an Gaussian kernel. Then use GridSearchCV to create a new estimator, that behaves exactly the same like a classifier. We add refit=True and choose verbose to whatever number we want, the higher the number, the more verbose (verbose just means the text output describing the process). So, we choose verbose=3 and fit the model for grid search. From this result we get the best estimator is SVC(C=10, gamma=0.01).
## 4. Build the Final Model
We build the final model and evaluate it with the opimal hyperparameters C=10 and gamma=0.01 by using the python code model= SVC(C=10, gamma=0.01, kernel=’rbf’). Then fit the model to the training dataset and create the confusion matrix to show the result. In this case we get the accuracy approximately as 96.677%. Then visualize the final model to the few pixel of traing data.
## 5. Visualize the Model in Test data
Now, we examine and plot the few pixels of test dataset. First, we read the test data and then make prediction on the test data. Then, we visualize the final model to some pixels of the test dataset and finally save the dataframe into a submission.csv file.

