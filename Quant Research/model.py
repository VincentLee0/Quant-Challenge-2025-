from abc import ABC, abstractmethod

class ResearchModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def fit(self, Xtrain, Ytrain, XValid=None, YValid=None):
        pass