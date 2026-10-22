
import numpy as np

def train_adaline(X, Label, LR, epochs, mse_threshold, bias):
    n_samples, n_features = X.shape
 
    w = np.random.uniform(-0.01, 0.01, n_features)
    b = np.random.uniform(-0.01, 0.01) if bias else 0.0
    mse_history = []
 
    # Repeat for m epochs
    for epoch in range(epochs):
 
        # For each training sample
        for i in range(n_samples):
            x = X[i]                    
            v = np.dot(w, x) + b        
            y = v                       
            error = Label[i] - y            
            w = w + LR * error * x      
            if bias:
                b = b + LR * error
 
        total = 0.0
        for i in range(n_samples):
            v = np.dot(w, X[i]) + b
            error = Label[i] - v
            total += 0.5 * error ** 2
        mse = total / n_samples
        mse_history.append(mse)
 
        if mse < mse_threshold:
            break
 
    return w, b, mse_history

def test_adaline(X, t, w, b):
    outputs = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x, ti in zip(X, t):
        v = np.dot(w, x) + b
        if v >= 0:
            y = 1
        else:
            y = -1

        if y == ti and y == 1:
            tp += 1
        elif y == ti and y == -1:
            tn += 1
        elif y != ti and y == 1:
            fp += 1
        else:
            fn += 1
        outputs.append(y)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    conf = np.array([[tn, fp],
                     [fn, tp]])
    return np.array(outputs), conf, accuracy