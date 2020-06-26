import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow import keras
import matplotlib.pyplt as plt
from os import listdir
from os.path import isfile, join
from shutil import *

path = 'audioset_v1_embeddings/'
bal = 'bal_train'

