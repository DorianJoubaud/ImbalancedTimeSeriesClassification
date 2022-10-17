
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from scipy.interpolate import CubicSpline
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from Models.resnet import *
from utils import *
from tensorflow.keras.utils import to_categorical
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def howmany(my_list, elt):
      tmp = 0
      for x in my_list:
                if (x == elt):
                          tmp += 1
      return tmp

def KL(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def take_sample(x,y, nb, mclass):
  res_x = list()
  res_y = list()
  for label in mclass:
    for i in range(nb):
      tmp = np.where(y == label)[0]

      tmp2 = random.choice(tmp)

      res_x.append(x[tmp2])
      res_y.append(y[tmp2])

  return res_x, res_y

def min_classes(d, e):
        """
        Calculates the number of minority classes. We call minority class to
        those classes with a probability lower than the equiprobability term.

        Parameters
        ----------
        _d : list of float.
            Empirical distribution of class probabilities.
        _e : float.
            Equiprobability term (1/K, where K is the number of classes).

        Returns
        -------
        Number of minority clases.
        """
        return len(d[d < e])

def raw_data(dataset, x_train, y_train, x_val, y_val, x_test, y_test, input_shape, nb_classes):

    model = RESNET('resnet/Raw/'+dataset, input_shape, nb_classes, False)
    model.build_model(input_shape, nb_classes)
    model.compile()
    y_raw = to_categorical(class_offset(y_train, dataset), nb_classes)

    histo = model.fit(x_train, y_raw, x_val, to_categorical( class_offset(y_val, dataset), nb_classes))
    y_pred = model.predict(x_test)

    f = f1_score(y_test, y_pred, average = None).tolist()
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None).tolist()
    pres = precision_score(y_test, y_pred, average=None).tolist()
    g = geometric_mean_score(y_test, y_pred, average=None).tolist()
    return accu, mcc, f, rec, pres, g, histo


def ROS_test(dataset, x_train, y_train, x_test,  y_test, input_shape,  nb_classes, sp_str):
    #Don t forget to put y train & y test to categorical

    oversample = RandomOverSampler(sampling_strategy=sp_str)
    X_over, y_over = oversample.fit_resample(x_train[:,:,0], y_train)
    model = RESNET('resnet/ROS/', input_shape, nb_classes, False)
    model.build_model(input_shape, nb_classes)
    y_over = to_categorical( class_offset(y_over, dataset), nb_classes)
    histo = model.fit(X_over, y_over)



    y_pred = model.predict(x_test)

    f = f1_score(y_test, y_pred, average = None).tolist()
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None).tolist()
    pres = precision_score(y_test, y_pred, average=None).tolist()
    g = geometric_mean_score(y_test, y_pred, average=None).tolist()
    return accu, mcc, f, rec, pres, g, histo






def jitter_test(dataset, x_train, y_train, x_test,  y_test, input_shape,  nb_classes,sp_str):

    def Augmentation(function, data, label_data, class_under, nb):
      underReprClass = list()
      idxs = np.where((label_data == class_under))

      count = 0


      for i in range(nb):


          if (count >= nb):

            break




          underReprClass.append(function(data[idxs[0][i%len(idxs)]]))
          count +=1

      return (np.array(underReprClass), np.array([class_under for i in range(nb)]))

    def jitter(x, sigma=0.03):
        # https://arxiv.org/pdf/1706.00527.pdf
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

    tilt = 0
    _, rat = np.unique(y_train, return_counts=True)


    for i in range(len(sp_str)):

          if (sp_str[i] - rat[i] != 0):



            if tilt == 0:
              aug = np.array([[[]]])
              aug_labels = list([[[]]])


              aug, aug_labels = Augmentation(jitter,x_train, y_train,i ,sp_str[i] - rat[i])


              tilt = 1
            else:

              tmp_data, tmp_labels = Augmentation(jitter,x_train, y_train,i ,sp_str[i] - rat[i])

              aug = np.concatenate((aug,tmp_data))
              aug_labels = np.concatenate((aug_labels, tmp_labels))


    oversamp = np.concatenate((x_train,aug), axis = 0)

    oversamp_labels = np.concatenate((y_train,aug_labels), axis = 0)
    oversamp_labels = to_categorical( class_offset(oversamp_labels, dataset), nb_classes)
    model = RESNET('resnet/Jitter/', input_shape, nb_classes, False)
    model.build_model(input_shape, nb_classes)
    y_over = to_categorical( class_offset(aug_labels, dataset), nb_classes)
    model.fit(aug, y_over)



    y_pred = model.predict(x_test)

    f = f1_score(y_test, y_pred, average = None).tolist()
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None).tolist()
    pres = precision_score(y_test, y_pred, average=None).tolist()
    g = geometric_mean_score(y_test, y_pred, average=None).tolist()
    return accu, mcc, f, rec, pres, g

def tw_test(dataset, x_train, y_train, x_test,  y_test, input_shape,  nb_classes,sp_str):
    def Augmentation(function, data, label_data, class_under, nb):
      underReprClass = list()
      idxs = np.where((label_data == class_under))[0]
      #print(idxs)

      count = 0


      for i in range(nb):


          if (count >= nb):

            break



          underReprClass.append(function(data)[idxs[i%len(idxs)]])

          count +=1

      return (np.array(underReprClass), np.array([class_under for i in range(nb)]))
    def time_warp(x, sigma=0.2, knot=4):

        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2,
                                                                    x.shape[2]))
        warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                scale = (x.shape[1]-1)/time_warp[-1]
                ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
        return ret
    aug = np.array([[[]]])
    aug_labels = list([[[]]])
    tilt = 0
    _, rat = np.unique(y_train, return_counts=True)


    for i in range(len(sp_str)):

          if (sp_str[i] - rat[i] != 0):


            if tilt == 0:


              aug, aug_labels = Augmentation(time_warp,x_train, y_train,i ,sp_str[i] - rat[i])


              tilt = 1
            else:

              tmp_data, tmp_labels = Augmentation(time_warp,x_train, y_train,i ,sp_str[i] - rat[i])

              aug = np.concatenate((aug,tmp_data))
              aug_labels = np.concatenate((aug_labels, tmp_labels))


    oversamp = np.concatenate((x_train,aug), axis = 0)

    oversamp_labels = np.concatenate((y_train,aug_labels), axis = 0)
    oversamp_labels = to_categorical( class_offset(oversamp_labels, dataset), nb_classes)
    model = RESNET('resnet/TW/', input_shape, nb_classes, False)
    model.build_model(input_shape, nb_classes)
    y_over = to_categorical( class_offset(aug_labels, dataset), nb_classes)
    model.fit(aug, y_over)



    y_pred = model.predict(x_test)

    f = f1_score(y_test, y_pred, average = None).tolist()
    accu = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average=None).tolist()
    pres = precision_score(y_test, y_pred, average=None).tolist()
    g = geometric_mean_score(y_test, y_pred, average=None).tolist()
    return accu, mcc, f, rec, pres, g



def SMOTE_test(dataset, x_train, y_train, x_val, y_val, x_test, y_test, input_shape, nb_classes, sp_st='all'):
     oversample = SMOTE(k_neighbors=1, sampling_strategy=sp_st)
     try:
        Xo, yo = oversample.fit_resample(x_train[:,:,0], y_train)
     except:
        try:
            oversample = SMOTE(k_neighbors=2)
            Xo, yo = oversample.fit_resample(x_train[:,:,0], y_train)
        except:
            return 0,0, np.zeros(nb_classes),np.zeros(nb_classes),np.zeros(nb_classes),np.zeros(nb_classes),0



     model = RESNET('resnet/SMOTE', input_shape, nb_classes, False)
     model.build_model(input_shape, nb_classes)
     y_over = to_categorical( class_offset(yo, dataset), nb_classes)
     histo = model.fit(Xo,y_over, x_val, to_categorical( class_offset(y_val, dataset), nb_classes))
     y_pred = model.predict(x_test)

     f = f1_score(y_test, y_pred, average = None).tolist()
     accu = accuracy_score(y_test, y_pred)
     mcc = matthews_corrcoef(y_test, y_pred)
     rec = recall_score(y_test, y_pred, average=None).tolist()
     pres = precision_score(y_test, y_pred, average=None).tolist()
     g = geometric_mean_score(y_test, y_pred, average=None).tolist()
     return accu, mcc, f, rec, pres, g,histo

def ADASYN_test(dataset, x_train, y_train, x_test,  y_test, input_shape,  nb_classes, sp_str = 'all'):
     oversample = ADASYN(sampling_strategy=sp_str)
     try:
         Xo, yo = oversample.fit_resample(x_train[:,:,0], y_train)
     except:
         print("ADASYN not possible, using SMOTE instead")
         try:
            oversample = SMOTE(k_neighbors=1)
            Xo, yo = oversample.fit_resample(x_train[:,:,0], y_train)
         except:
            return 0,0, np.zeros(nb_classes),np.zeros(nb_classes),np.zeros(nb_classes),np.zeros(nb_classes)

     else:
         Xo, yo = oversample.fit_resample(x_train[:,:,0], y_train)



     model = RESNET('resnet/ADASYN/', input_shape, nb_classes, False)
     model.build_model(input_shape, nb_classes)
     y_over = to_categorical( class_offset(yo, dataset), nb_classes)
     model.fit(Xo, y_over)
     y_pred = model.predict(x_test)

     f = f1_score(y_test, y_pred, average = None).tolist()
     accu = accuracy_score(y_test, y_pred)
     mcc = matthews_corrcoef(y_test, y_pred)
     rec = recall_score(y_test, y_pred, average=None).tolist()
     pres = precision_score(y_test, y_pred, average=None).tolist()
     g = geometric_mean_score(y_test, y_pred, average=None).tolist()
     return accu, mcc, f, rec, pres, g


