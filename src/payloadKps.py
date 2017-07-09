import os
import pickle


class payloadKps:
    def __init__(self, myDesc, myKps):
        self.myDesc = myDesc
        self.myKps = myKps


def buildPicklePath(path):
    return path[:path.rfind('.')] + '.pickle'

def savePickle(p, path):
    with open(buildPicklePath(path), 'wb') as f:
        pickle.dump(p, f)

def loadPickle(path):
    with open(buildPicklePath(path), 'rb') as f:
        p = pickle.load(f)
        return p.myDesc, p.myKps

def pickleExist(path):
    return os.path.isfile(buildPicklePath(path))
