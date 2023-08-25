from model.train_test import feature_extraction, model

'''Building predictive system with message from mail.csv from data folder'''
input_mail = ["England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 "
              "Try:WALES, SCOTLAND 4txt/ú1.20 POBOXox36504W45WQ 16+"]

# convert text to feature vectors

input_data_feature = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_feature)
print(prediction)

# check output value of prediction

# my_list = [1, 2, 3]
#
# if prediction[0] == 1:
#     print(my_list[0])

if prediction[0] == 1:
    print('Mail spam')
else:
    print('Mail ham')

