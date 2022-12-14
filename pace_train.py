from utils.dataloader import NpzLoader
from models.pacemodel import PaceNetwork
from torch.utils.tensorboard import SummaryWriter


# TODO remove rotation 1， 6， 17， 24
if __name__ == '__main__':
    file_train = 'h36m_train_new.npz'
    npzloader_train = NpzLoader(file_train, visualize=False, window=50, offset=0)
    file_val = 'h36m_test_new.npz'
    npzloader_val = NpzLoader(file_val, window=26, offset=0, visualize=False)
    print('load complete')
    pacenet = PaceNetwork()
    writer = SummaryWriter('log')
    pacenet.train(npzloader_train, npzloader_val, 60, 1, writer, window=50)
