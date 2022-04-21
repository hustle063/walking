from utils.dataloader import NpzLoader
from models.model import PhysNetwork

if __name__ == '__main__':
    model_file = 'E:/trained_model/h36m_2_min_560.t7'
    file = 'h36m_test.npz'
    npzloader = NpzLoader(file, visualize=False, offset=360)
    physnet = PhysNetwork()
    physnet.predict(npzloader, model_file, 4)
