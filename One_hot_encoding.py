import numpy as np
import Dictionary 

voc_size = Dictionary.vocabulary_size
vec_dim = voc_size + 2
total_words = Dictionary.total_words
id_list = list(range(1,voc_size+1))
max_length= 22

table = dict(zip(total_words,id_list))

def one_hot_encoding(word):
    temp = np.zeros((1,vec_dim))
    temp[0,table[word]] = 1
    return temp

def seq_encoding(seq):
    output = []
    valid_length = len(seq)
    if valid_length <= max_length-2:
       for w in range(max_length):
           temp = np.zeros((1,vec_dim))
           if w == 0:
              temp[0,0]=1
           elif w == max_length-1:
              temp[0,vec_dim-1]=1
           elif w>valid_length:
              temp = temp
           elif w>=1 and w<=valid_length:
              temp = one_hot_encoding(seq[w-1])
           output.append(temp)
       return output
    else:
       print('Sequence length error!')


