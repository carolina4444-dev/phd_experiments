from abc import ABC, abstractmethod


class NASProblem(ABC):

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def export_architecture(self):
        pass