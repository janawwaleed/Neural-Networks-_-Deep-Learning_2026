import numpy as np

class Perceptron:
    def __init__(self, learningRate, epochs, use_bias):
        self.learningRate = learningRate
        self.epochs = epochs
        self.use_bias = use_bias
        self.w=None


    def train(self, X, t):
        X = np.asarray(X, dtype=float)
        if self.use_bias:
            X = np.c_[np.ones(len(X)), X]
        self.w = np.random.rand(X.shape[1]) * 0.1
        for epoch in range (self.epochs):
            for xi,ti in zip(X,t):
                netVal=np.dot(xi, self.w)
                if netVal>=0:
                    y=1
                else:
                    y=-1
                if y!=ti:
                    e=ti-y
                    self.w = self.w + self.learningRate * e * xi   


    def test(self, X, t):
        X = np.asarray(X, dtype=float)
        if self.use_bias:
            X = np.c_[np.ones(len(X)), X]
        tp= 0       # (true +1, predicted +1) true pos
        tn = 0      # (true -1, predicted -1) true neg
        fp = 0      # (true -1, predicted +1) false pos
        fn = 0      # (true +1, predicted -1)  false neg
      
        for xi,ti in zip(X,t):
            netVal=np.dot(xi, self.w)
            if netVal>=0:
                y=1
            else:
                y=-1
            if y==ti and y==1:
                tp+=1   
            elif y==ti and y==-1: 
                tn+=1
            elif y!=ti and y==1: 
                fp+=1
            else:
                fn+=1
        total=tp+tn+fp+fn    
        accuracy=(tp+tn)/total
        conf = np.array([[tn, fp],            
                        [fn, tp]])
        return conf,accuracy

               
        