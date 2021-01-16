import numpy as np
# np.seterr(divide='ignore', invalid='ignore')


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 1

    def optimize(self, w, dw):
        """
        w: weight hien tai
        dw: gradient descent (dao ham loss theo w)
        """
        self.m = self.beta1*self.m+(1-self.beta1)*dw
        self.v = self.beta2*self.v+(1-self.beta2)*(dw**2)

        m_corr = self.m/(1-self.beta1**self.t)

        v_corr = self.v/(1-self.beta2**self.t)

        w = w - self.learning_rate*(m_corr/(np.sqrt(v_corr)+self.epsilon))

        self.t += 1
        return w
