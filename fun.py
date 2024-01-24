from torch import tensor
import random

def one_hot_max(prediction_t):
    oh_prediction = [0.]*len(prediction_t)
    index = (prediction_t.tolist()).index(max(prediction_t))
    oh_prediction[index] = 1.
    oh_prediction_t = tensor(oh_prediction)
    return oh_prediction_t
def calc_accuracy(prediction_t,realite_t):
    nb_bon = 0
    for i in range(len(prediction_t)):
        if prediction_t[i] == realite_t[i]:
            nb_bon +=1
    accuracy = nb_bon/len(prediction_t)
    return accuracy
def train_test(list_label_t,list_image_t):
    seed_value = random.randint(1,1000)
    random.seed(seed_value)
    random.shuffle(list_label_t)
    random.seed(seed_value)
    random.shuffle(list_image_t)
    train_list_label_t , test_list_label_t = list_label_t[50:] , list_label_t[:50]
    train_list_image_t , test_list_image_t = list_image_t[50:] , list_image_t[:50]
    return train_list_label_t , test_list_label_t , train_list_image_t , test_list_image_t