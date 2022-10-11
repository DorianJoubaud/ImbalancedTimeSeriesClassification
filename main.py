

from utils import *
from tensorflow.keras.utils import to_categorical
from functions import *
from Models.resnet import RESNET
import numpy as np
import os
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split





bench = list()

#balance = pd.read_csv('balance_measure.csv') #We use Shanon Entropy as reference

folders = os.listdir('data')




accuracy = list()
mathc = list()
f1 = list()
gm = list()
presi = list()
reca = list()


for i in range(len(folders)):
    tmp_bench = list()
    dataset = folders[i]

    print(dataset)
    print(f'{i}/{len(folders)}')

    nb_class = nb_classes(dataset)
    nb_dims = nb_dims(dataset)




    data, labels = get_data(dataset, '\t')
    labels = class_offset(labels, dataset)
    x_train, x_rem, y_train, y_rem = train_test_split(data,labels, train_size=0.8)
    x_val, x_test, y_val, y_test = train_test_split(x_rem,y_rem, test_size=0.5)

    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)





    x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))
    y_test = to_categorical(class_offset(y_test, dataset), nb_class)

    _, rat = np.unique(y_train, return_counts=True)
    majority_class = np.argmax(rat)

    sp_str = {i: rat[majority_class] for i in range(len(rat))}







    evolutionf = list() #f1 scores
    evolutiong = list() #g means
    evolutiona = list() #accuracy
    evolutionmcc = list() #mcc
    evolutionrec = list() #precision
    evolutionrec = list() #precision
    evolutionpres = list() #recall

    #RAW DATA

    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])
    histo = list()

    for i in range(1):
        taccu,tmcc, tf, trec, tpres, tg, histo = raw_data(dataset, x_train, y_train,x_val, y_val, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class)

        accu += taccu
        mcc += tmcc
        f += tf

        rec += trec
        pres += tpres
        g += tg

    evolutionf.append(f/1) # f1 scores
    evolutiong.append(g/1) #g means
    evolutiona.append(accu/1) #accuracy
    evolutionmcc.append(mcc/1) #mcc
    evolutionrec.append(rec/1) #precision
    evolutionpres.append(pres/1) #recall

    """     #ROS

    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])

    for i in range(3):
        taccu,tmcc, tf, trec, tpres, tg = ROS_test(dataset, x_train, y_train, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_str)

        accu += taccu
        mcc += tmcc
        f += tf

        rec += trec
        pres += tpres
        g += tg

    evolutionf.append(f/3) # f1 scores
    evolutiong.append(g/3) #g means
    evolutiona.append(accu/3) #accuracy
    evolutionmcc.append(mcc/3) #mcc
    evolutionrec.append(rec/3) #precision
    evolutionpres.append(pres/3) #recall

    #Jittering
    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])

    for i in range(3):
        taccu,tmcc, tf, trec, tpres, tg = jitter_test(dataset, x_train, y_train, x_test, np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_str)

        accu += taccu
        mcc += tmcc
        f += tf

        rec += trec
        pres += tpres
        g += tg

    evolutionf.append(f/3) # f1 scores
    evolutiong.append(g/3) #g means
    evolutiona.append(accu/3) #accuracy
    evolutionmcc.append(mcc/3) #mcc
    evolutionrec.append(rec/3) #precision
    evolutionpres.append(pres/3) #recall

    #TW

    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])

    for i in range(3):
        taccu,tmcc, tf, trec, tpres, tg = tw_test(dataset, x_train, y_train, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_str)

        accu += taccu
        mcc += tmcc
        f += tf

        rec += trec
        pres += tpres
        g += tg

    evolutionf.append(f/3) # f1 scores
    evolutiong.append(g/3) #g means
    evolutiona.append(accu/3) #accuracy
    evolutionmcc.append(mcc/3) #mcc
    evolutionrec.append(rec/3) #precision
    evolutionpres.append(pres/3) #recall


    #SMOTE
    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])

    for i in range(3):
        taccu,tmcc, tf, trec, tpres, tg = SMOTE_test(dataset, x_train, y_train, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_str)

        accu += taccu
        mcc += tmcc
        f += tf

        rec += trec
        pres += tpres
        g += tg

    evolutionf.append(f/3) # f1 scores
    evolutiong.append(g/3) #g means
    evolutiona.append(accu/3) #accuracy
    evolutionmcc.append(mcc/3) #mcc
    evolutionrec.append(rec/3) #precision
    evolutionpres.append(pres/3) #recall


    #ADASYN
    accu = 0
    mcc = 0
    f = np.array([0.0 for cl in range(nb_class)])
    rec = np.array([0.0 for cl in range(nb_class)])
    pres = np.array([0.0 for cl in range(nb_class)])
    g = np.array([0.0 for cl in range(nb_class)])

    for i in range(3):
        taccu,tmcc, tf, trec, tpres, tg = ADASYN_test(dataset, x_train, y_train, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_str)

        accu += taccu
        mcc += tmcc
        f += tf

        rec += trec
        pres += tpres
        g += tg

    evolutionf.append(f/3) # f1 scores
    evolutiong.append(g/3) #g means
    evolutiona.append(accu/3) #accuracy
    evolutionmcc.append(mcc/3) #mcc
    evolutionrec.append(rec/3) #precision
    evolutionpres.append(pres/3) #recall


 """
    print(f'folders[i]', evolutionf)


    accuracy.append(evolutiona)
    mathc.append(evolutionmcc)
    f1.append(evolutionf)
    gm.append(evolutiong)
    presi.append(evolutionpres)
    reca.append(evolutionrec)

    print(f1)










os.makedirs('Results/Accuracy', exist_ok=True)
os.makedirs('Results/MCC', exist_ok=True)
os.makedirs('Results/F1_scores', exist_ok=True)
os.makedirs('Results/G_scores', exist_ok=True)
os.makedirs('Results/Precision', exist_ok=True)
os.makedirs('Results/Recall', exist_ok=True)
os.makedirs('Results/Historic', exist_ok=True)


hist = pd.DataFrame(histo)
hist.to_csv('Results/Historic/hist.csv')


acc = pd.DataFrame(accuracy)

acc = acc.rename(columns={0 : 'Raw',1 : 'ROS', 2:'Jittering', 3:'Time Warping', 4:'SMOTE', 5:'ADASYN'})
acc = acc.rename(index={i : folders[i] for i in range(len(folders))})
print('---------------------------------------')
print(acc)
acc.to_csv('Results/Accuracy/acc.csv')

mcc = pd.DataFrame(mathc)

mcc = mcc.rename(columns={0 : 'Raw',1 : 'ROS', 2:'Jittering', 3:'Time Warping', 4:'SMOTE', 5:'ADASYN'})
mcc = mcc.rename(index={i : folders[i] for i in range(len(folders))})
print(mcc)
mcc.to_csv('Results/MCC/mcc.csv')


f = pd.DataFrame(f1)

f= f.rename(columns={0 : 'Raw',1 : 'ROS', 2:'Jittering', 3:'Time Warping', 4:'SMOTE', 5:'ADASYN'})
f = f.rename(index={i : folders[i] for i in range(len(folders))})
print(f)
f.to_csv('Results/F1_scores/f1.csv')

g = pd.DataFrame(gm)

g = g.rename(columns={0 : 'Raw',1 : 'ROS', 2:'Jittering', 3:'Time Warping', 4:'SMOTE', 5:'ADASYN'})
g = g.rename(index={i : folders[i] for i in range(len(folders))})
print(g)
g.to_csv('Results/G_scores/g.csv')

p= pd.DataFrame(presi)

p= p.rename(columns={0 : 'Raw',1 : 'ROS', 2:'Jittering', 3:'Time Warping', 4:'SMOTE', 5:'ADASYN'})
p = p.rename(index={i : folders[i] for i in range(len(folders))})
print(p)
p.to_csv('Results/Precision/pres.csv')

r= pd.DataFrame(recall)

r= r.rename(columns={0 : 'Raw',1 : 'ROS', 2:'Jittering', 3:'Time Warping', 4:'SMOTE', 5:'ADASYN'})
r = r.rename(index={i : folders[i] for i in range(len(folders))})
r.to_csv('Results/Recall/rec.csv')

