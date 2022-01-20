from utils.dataloader import NpzLoader
from models.model import PhysNetwork
from torch.utils.tensorboard import SummaryWriter


# TODO remove rotation 1， 6， 17， 24
if __name__ == '__main__':
    file_train = 'edin_train_less_30fps.npz'
    npzloader_train = NpzLoader(file_train, visualize=False)
    file_val = 'edin_test_less_30fps.npz'
    npzloader_val = NpzLoader(file_val, window=21, offset=180, visualize=False)
    print('load complete')
    physnet = PhysNetwork()
    writer = SummaryWriter('log')
    physnet.train(npzloader_train, npzloader_val, 120, 32, writer)


