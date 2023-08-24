import numpy as np
import pandas as pd
from Src.config import raw_mail_data

#
# '''loading the data '''
# print(raw_mail_data)

'''replace the mail values with a null string'''
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

'''printing the first 5 rows of the dataframe'''
print(mail_data.head())

'''checking the nr of rows and columns in the dataframe'''
print(mail_data.shape)

'''Label enconding'''
# label spam null as 0, ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# spam = 0 and ham = 1
'''separating data as text and label'''
X = mail_data['Message']

Y = mail_data['Category']
print(X)  # message
print(Y)  # data 1 or 0



