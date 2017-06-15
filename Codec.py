import SPEC
import numpy as np
class Codec:
    def __init__(self):
        pass
    encode_map = dict(zip(SPEC.all_words,range(SPEC.vocabulary)))
    decode_action_map = dict(zip(range(len(SPEC.home_actions)),SPEC.home_actions))
    decode_object_map = dict(zip(range(len(SPEC.home_objects)),SPEC.home_objects))
    def encode(self,raw_str):
        code = [self.encode_map[word] for line in raw_str.split(".\n") if len(line)!=0 for word in [(0,)]+line.split(" ")+["."]]
        code += [0] * (SPEC.seq_len * SPEC.seq_num)
        return (np.eye(SPEC.vocabulary)[code]).astype(np.int32)
    def decode_action(self,a,o):
        action_strs = self.decode_action_map[a]
        object_strs = self.decode_object_map[o]
        return action_strs,object_strs