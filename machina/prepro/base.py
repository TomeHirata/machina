import numpy as np
import torch

class BasePrePro(object):
    def __init__(self, ob_space, normalize_ob=True):
        self.ob_space = ob_space
        self.normalize_ob = normalize_ob
        if self.normalize_ob:
            self.ob_rm = np.zeros(self.ob_space.shape)
            self.ob_rv = np.ones(self.ob_space.shape)
            self.alpha = 0.001

    def update_ob_rms(self, ob):
        self.ob_rm = self.ob_rm * (1-self.alpha) + self.alpha * ob
        self.ob_rv = self.ob_rv * (1-self.alpha) + self.alpha * np.square(ob-self.ob_rm)

    def prepro(self, ob):
        if self.normalize_ob:
            ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
            ob = np.clip(ob, -5, 5)
        return ob

    def prepro_with_update(self, ob):
        if self.normalize_ob:
            ob = (ob - self.ob_rm) / (np.sqrt(self.ob_rv) + 1e-8)
            self.update_ob_rms(ob)
            ob = np.clip(ob, -5, 5)
        return ob

