from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from Src.processing import X, Y

'''splitting the data into training data and test data'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)  # total number
print(X_test.shape)  # total number

'''Feature expection'''
# transforms the text data to feature vectors that can be used to the Logistic regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')

X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

'''convert Y_train and Y_test values as integer'''

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
