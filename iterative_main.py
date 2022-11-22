
from utils import *
from tensorflow.keras.utils import to_categorical
from functions import *
from Models.resnet import RESNET
import numpy as np
import os
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split


#get datasets names
folders = os.listdir('data')


# for each dataset
for i in range(len(folders)):

    tmp_bench = list()
    dataset = folders[i]

    print(dataset)
    print(f'{i}/{len(folders)}')

    nb_class = nb_classes(dataset)
    nb_dims = 1



    #get data
    x_train,x_test,y_train,y_test = get_data(dataset, '\t')
    #Split train val test
    y_train = class_offset(y_train, dataset)
    y_test = class_offset(y_test, dataset)
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test, test_size=0.5)


    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    #normalise in [-1;1]
    x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
    x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

    nb_timesteps = int(x_train.shape[1] / nb_dims)
    input_shape = (nb_timesteps , nb_dims)




    #reshape 
    x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
    x_val = x_val.reshape((-1, input_shape[0], input_shape[1]))
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

    y_test = to_categorical(class_offset(y_test, dataset), nb_class)

    #Get class distribution
    _, rat = np.unique(y_train, return_counts=True)
    
    y_majority_idxs = list() #majority class
    majority_class = np.argmax(rat)
    for label in range(len(y_train)):
      if (rat[int(y_train[label])] == rat[majority_class]):
        y_majority_idxs.append(label)


    y_majority_idxs = np.array(y_majority_idxs)
    print(y_majority_idxs)
    y_non_majority_idxs = np.where(np.array(rat) != rat[majority_class])[0].tolist()


    x_train_new = x_train[y_majority_idxs]
    y_train_new = y_train[y_majority_idxs]
    labels = y_train_new.tolist()

    #Reduce initial data to max imbalance 
    #Exemple : (4,8,6) -> (2,8,2)
    #We keep 2 element for minority classes in order to apply SMOTE etc
    x_mino,y_mino = take_sample(x_train,y_train, 2, y_non_majority_idxs)

    x_train_new = np.concatenate((x_train_new, np.array(x_mino)))

    y_train_new = np.concatenate((y_train_new, y_mino))

    _, rat_beg = np.unique(y_train_new, return_counts=True)
    
    e = [rat[majority_class] for _ in range(nb_class)]
    
    #variables that will store our results
    df_acc = []
    df_mcc = []
    df_f1 = []
    df_g = []
    df_pres = []
    df_rec = []
    df_KL = []
    
    
    taccu, tmcc, tf, trec, tpres, tg   = raw_data(dataset, x_train_new, y_train_new, x_val, y_val, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class)
    
    df_acc.append([0,taccu, taccu, taccu, taccu, taccu])
    df_KL.append(KL(e, rat_beg))
    
    total_to_add = nb_class * rat_beg[majority_class] - rat_beg.sum() #nb of element to add to balance
    sorted_rat_next = np.argsort(rat_beg)
    
    nb_add = 0
    
    while(nb_add != total_to_add):

      for i in sorted_rat_next:

        if (rat_beg[i] != rat_beg[majority_class]):

          labels.append(i)
          _, tmp_rat = np.unique(labels, return_counts=True)




          df_KL.append(KL(e,tmp_rat))
          print(tmp_rat)
          sp_strg = {i:tmp_rat[i] for i in range(len(tmp_rat))}
          
          
          raccu,rmcc, rf, rrec, rpres, rg   = ROS_test(dataset, x_train_new, y_train_new,x_val, y_val, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_strg)
          jaccu,jmcc, jf, jrec, jpres, jg   = jitter_test(dataset, x_train_new, y_train_new,x_val, y_val, x_test, np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_strg)
          taccu,tmcc, tf, trec, tpres, tg   = tw_test(dataset, x_train, y_train,x_val, y_val, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_strg)
          saccu,smcc, sf, srec, spres, sg   = SMOTE_test(dataset, x_train_new, y_train_new,x_val, y_val, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_strg)
          aaccu,amcc, af, arec, apres, ag   = ADASYN_test(dataset, x_train, y_train,x_val, y_val, x_test,  np.argmax(y_test, axis = 1), input_shape,  nb_class, sp_strg)
          
          nb_add += 1
          
          df_acc.append([nb_add, raccu, jaccu, taccu, saccu,aaccu])
          df_mcc.append([nb_add, rmcc, jmcc, tmcc, smcc, amcc])
          df_f1.append([nb_add, rf, jf, tf, sf, af])
          df_g.append([nb_add, rg, jg, tg, sg, ag])
          df_pres.append([nb_add, rpres, jpres, tpres, spres, apres])
          df_rec.append([nb_add, rrec, jrec, trec, srec, arec])
    pd.DataFrame(df_acc, columns={0 : 'ROS', 1:'Jittering', 2:'Time Warping', 3:'SMOTE', 4:'ADASYN'}).to_csv(f'Iterative_Results/{dataset}/acc.csv')
    pd.DataFrame(df_mcc, columns={0 : 'ROS', 1:'Jittering', 2:'Time Warping', 3:'SMOTE', 4:'ADASYN'}).to_csv(f'Iterative_Results/{dataset}/mcc.csv')
    pd.DataFrame(df_f1, columns={0 : 'ROS', 1:'Jittering', 2:'Time Warping', 3:'SMOTE', 4:'ADASYN'}).to_csv(f'Iterative_Results/{dataset}/f.csv')
    pd.DataFrame(df_g, columns={0 : 'ROS', 1:'Jittering', 2:'Time Warping', 3:'SMOTE', 4:'ADASYN'}).to_csv(f'Iterative_Results/{dataset}/g.csv')
    pd.DataFrame(df_pres, columns={0 : 'ROS', 1:'Jittering', 2:'Time Warping', 3:'SMOTE', 4:'ADASYN'}).to_csv(f'Iterative_Results/{dataset}/pres.csv')
    pd.DataFrame(df_rec, columns={0 : 'ROS', 1:'Jittering', 2:'Time Warping', 3:'SMOTE', 4:'ADASYN'}).to_csv(f'Iterative_Results/{dataset}/rec.csv')

    
    
    