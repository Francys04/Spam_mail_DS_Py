"""This line imports the train_test_split function, which is used to split a dataset into training and testing subsets.
 It's a common step in machine learning to evaluate models."""
from sklearn.model_selection import train_test_split
"""This line imports the TfidfVectorizer class, which is used for text feature extraction. 
It converts text data into numerical vectors based on 
Term Frequency-Inverse Document Frequency (TF-IDF) values, making it suitable for text-based machine learning tasks."""
from sklearn.feature_extraction.text import TfidfVectorizer
"""This line imports the LogisticRegression class, which is used for logistic regression classification. 
Logistic regression is a common algorithm for binary and multi-class classification tasks."""
from sklearn.linear_model import LogisticRegression
"""This line imports the accuracy_score function, which is used to calculate the accuracy of a machine learning 
model's predictions. It's a common metric for classification tasks."""
from sklearn.metrics import accuracy_score
from Src.processing import X, Y

'''splitting the data into training data and test data'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)  # total number
print(X_test.shape)  # total number

'''Feature expection'''
# transforms the text data to feature vectors that can be used to the Logistic regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', binary=True, lowercase=True)

X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

'''convert Y_train and Y_test values as integer'''

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
print(X_train)
print(X_train_feature)

'''
Training model
Logistic Regression 
'''
model = LogisticRegression()

# training the logistic Regression with the training data
model.fit(X_train_feature, Y_train)

'''Evaluating the trained model'''
# prediction on training data
prediction_on_training_data = model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Accuracy on training data : ", accuracy_on_training_data)

# prediction on test data
prediction_on_test_data = model.predict(X_test_feature)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print("Accuracy on test data : ", accuracy_on_test_data)

'''
Conclusion: not much different between trained data and test data
'''
