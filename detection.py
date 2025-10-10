import numpy as np
from scipy.fft import dct


def detection(image, watermarked, alpha, mark_size, v='multiplicative', freq_range='high'):
    ori_dct = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    wat_dct = dct(dct(watermarked, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Use absolute values for location selection only
    abs_dct = abs(ori_dct)
    locations = np.argsort(-abs_dct, axis=None)
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]
    
    # Select frequency range
    if freq_range == 'high':
        selected_locs = locations[1:mark_size + 1]
    elif freq_range == 'mid':
        selected_locs = locations[500:500 + mark_size]
    elif freq_range == 'low':
        selected_locs = locations[100:100 + mark_size]
    else:
        selected_locs = locations[1:mark_size + 1]
    
    w_ex = np.zeros(mark_size, dtype=np.float64)
    
    for idx, loc in enumerate(selected_locs):
        if v == 'additive':
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / alpha
        elif v == 'multiplicative':
            w_ex[idx] = (wat_dct[loc] - ori_dct[loc]) / (alpha * ori_dct[loc])
    
    return w_ex
   


def similarity(X, X_star):
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s


def compute_threshold(mark_size, w, N):
    SIM = np.zeros(N)
    for i in range(N):
        r = np.random.uniform(0.0, 1.0, mark_size)
        SIM[i] = similarity(w, r)
    SIMs = SIM.copy()
    SIM.sort()
    t = SIM[-1]
    T = t + (0.1 * t)
    return T, SIMs