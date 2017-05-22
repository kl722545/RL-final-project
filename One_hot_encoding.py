import numpy as np
import Dictionary 


## Vocabulary size ##################
voc_size = Dictionary.vocabulary_size
#####################################

## Vector dimension , including ##
## <SOS> and <EOS> ##
vec_dim = voc_size + 2
##################################

## Totol words list ################
total_words = Dictionary.total_words
####################################

## Tuple list #####################
id_list = list(range(1,voc_size+1))
###################################

## Maximum length of sequence #####
max_length= 22
###################################

##### Word to encoding tuple table #####
table = dict(zip(total_words,id_list))
########################################

##### One-hot encoding ############# 
## Convert word to one-hot vector ##
def one_hot_encoding(word):
    temp = np.zeros((1,vec_dim))
    temp[0,table[word]] = 1
    return temp
####################################

##### Sequence encoding ############################
## Convert sequence to several vectors ##
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
#####################################################

