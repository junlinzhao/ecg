from sklearn import preprocessing
import pywt
import numpy as np
from tqdm import tqdm
import warnings


class Normalizer(object):
    def __init__(self, strategy):
        self.scaler = None
        self.strategy = strategy

    def _dim_fix(self, x):
        if (len(x.shape) == 2):
            x = np.expand_dims(x, axis=-1)
        assert(len(x.shape) == 3)
        return x

    def fit(self, x):
        print('Fitting Normalization: ' + self.strategy)
        x = self._dim_fix(x)
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        if self.strategy == 'standard_scale':
            self.scaler = preprocessing.StandardScaler().fit(x)
        elif self.strategy == 'min_max':
            self.scaler = preprocessing.MinMaxScaler(
                feature_range=(-1, 1)).fit(x)
        elif self.strategy == 'robust_scale':
            self.scaler = preprocessing.RobustScaler().fit(x)
        else:
            raise ValueError("Strategy not found!")

    def transform(self, x):
        print('Applying Normalization...')
        x = self._dim_fix(x)
        original_shape = x.shape
        new_shape = (x.shape[0]*x.shape[1], x.shape[2])
        return self.scaler.transform(
            x.reshape(new_shape)).reshape(original_shape)


EXPECTED_TRANSFORMED_LENGTH = 3000


class DiscreteWaveletTransformer(object):
    def __init__(self, wavelet_fns):
        self.transforms = wavelet_fns

    def transform(self, x):
        print('Applying Wavelet Transformations...')
        x_new = []
        for x_indiv in tqdm(x):
            x_indiv_trans = []
            for wavefn in self.transforms:
                transform = np.array(pywt.dwt(
                    x_indiv, wavefn, mode='constant'))
                if(transform.shape[1] != EXPECTED_TRANSFORMED_LENGTH):
                    warnings.warn(
                        "Reshaping to proper length after wavelet transform")
                    transform = transform[:, :EXPECTED_TRANSFORMED_LENGTH]
                x_indiv_trans.extend(transform)
            x_new.append(np.array(x_indiv_trans).T)
        x_new = np.array(x_new)
        return x_new