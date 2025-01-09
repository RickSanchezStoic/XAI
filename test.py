from data import getBreastData
train_data, val_data, test_data = getBreastData()

from model.linear import generate_model
model = generate_model(30,2,[100,50])


from model.training import train_model

history = train_model(model,train_data,val_data,'Adam','cross_entropy',device='cpu')
# for i in range(0, len(train_data[0]), 1):
#     inputs = train_data[0][i:i + 1]
#     labels = train_data[1][i:i + 1]
#     print(inputs, labels, "\n\n\n")

import captum


