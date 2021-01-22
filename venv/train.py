import tensorflow as tf
import model,get_pic
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
data_url = r'C:\Users\14999\Desktop\data'
label_url = r'C:\Users\14999\Desktop\label'
row = 128
col = 128
dataSet,labelSet = get_pic.creat_x_database()
# x_train,x_test,y_train,y_test = train_test_split(dataSet,labelSet,test_size=0.2,random_state=0)
# x_train = x_train/1.0
# x_test = x_test/255.0
# y_train = y_train/1.0
# y_test = y_test/255.0
while True:
    myModel = tf.keras.models.load_model(r'C:\Users\14999\Desktop\model')
    # myModel = model.unet()
    batch_size = 1
    epoches = 1
    train_num = len(dataSet)*0.8
    test_num = len(labelSet)*0.2
    modelcheck = tf.keras.callbacks.ModelCheckpoint(r'C:\Users\14999\Desktop\model',monitor='val_accuracy',save_best_only=True,mode='max')
    callable = [modelcheck]
    index = [i for i in range(len(dataSet))]
    np.random.shuffle(index)
    dataSet = dataSet[index]
    labelSet = labelSet[index]
    # print(labelSet[0])
    labelSet1 = labelSet/255.0
    dataSet1 = dataSet/255.0
    H = myModel.fit(dataSet1,labelSet1,steps_per_epoch=train_num//batch_size,epochs=epoches,verbose=1,
                        validation_split=0.1,callbacks=callable,max_queue_size=1,shuffle=True)
# P = myModel.predict()
plt.style.use("ggplot")
plt.figure()
N = epoches
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(r'C:\Users\14999\Desktop\plot')