from utils.dataloader import NpzLoader
from models.model import PhysNetwork

if __name__ == '__main__':
    model_file = 'E:/trained_model/checkpoint36_min_167.t7'
    file = 'edin_test_less_30fps.npz'
    npzloader = NpzLoader(file, visualize=False, offset=360)
    physnet = PhysNetwork()
    physnet.predict(npzloader, model_file, 4)
