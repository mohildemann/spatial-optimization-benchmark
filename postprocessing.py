import numpy as np

class Solution:
    _id = 0

    def __init__(self, representation):
        self._solution_id = Solution._id
        Solution._id += 1
        self.representation = representation

def load_patch_ID_map(filepath):
    return np.genfromtxt(filepath, dtype=int, skip_header=6, filling_values='-1')

def read_patch_ID_map(patchmap, solution_represenation, static_element,No_Data_Value = 0, input_patch_map = None):
    if input_patch_map is None:
        patches = load_patch_ID_map(patchmap)
    else:
        patches = input_patch_map
    landuseraster = []
    counter = 0
    for rowid in range(patches.shape[0]):
        colvalues = []
        for i in range(patches.shape[1]):
            if patches[rowid,i] == No_Data_Value:
                colvalues.append(static_element)
            else:
                colvalues.append(solution_represenation[counter])
                counter += 1

        landuseraster.append(colvalues)
    reversed_lum= np.flip(np.array(landuseraster),axis=0)
    return reversed_lum