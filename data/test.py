import pickle
import numpy as np

with open("Manhattan_dic.pkl",'rb') as f:
    data = pickle.load(f)
print(data['centroid_lat'])
print(data['centroid_lon'])
print(data['zone_num'])
print(data['map'])

print(np.min(data['centroid_lat']))
print(np.max(data['centroid_lat']))

print(np.min(data['centroid_lon']))
print(np.max(data['centroid_lon']))