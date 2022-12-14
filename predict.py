from utils.dataloader import NpzLoader
from models.model import PhysNetwork
from models.pacemodel import PaceNetwork

if __name__ == '__main__':
    model_file = 'C:/Users/yuchhuang9/walking/trained_model/h36m_pace_28_final.t7'
    file = 'h36m_test.npz'
    npzloader = NpzLoader(file, visualize=False, offset=0, window=75)
    # physnet = PhysNetwork()
    # physnet.predict(npzloader, model_file, 4)
    pacenet = PaceNetwork()
    pacenet.predict(npzloader, model_file, 8, window=75)
