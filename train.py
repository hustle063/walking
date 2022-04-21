from utils.dataloader import NpzLoader
from models.model import PhysNetwork
from torch.utils.tensorboard import SummaryWriter


# TODO remove rotation 1， 6， 17， 24
if __name__ == '__main__':
    file_train = 'h36m_train.npz'
    npzloader_train = NpzLoader(file_train, visualize=False, window=50, offset=-25)
    file_val = 'h36m_test.npz'
    npzloader_val = NpzLoader(file_val, window=26, offset=0, visualize=False)
    print('load complete')
    physnet = PhysNetwork()
    writer = SummaryWriter('log')
    physnet.train(npzloader_train, npzloader_val, 120, 32, writer)


