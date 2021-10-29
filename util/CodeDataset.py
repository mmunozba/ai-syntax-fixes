from torch.utils.data import Dataset


class CodeDataset(Dataset):
    # load the dataset
    def __init__(self, X, Y):
        # store the inputs and outputs

        self.X = X
        self.y = Y
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]