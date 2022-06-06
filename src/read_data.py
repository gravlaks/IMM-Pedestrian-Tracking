import scipy.io

def read_data(filepath="data/uah2.mat"):
    mat = scipy.io.loadmat(filepath)
    GT = mat["GT"]
    Z = mat["TD"]

    T0 = GT[0, 0]
    GT[:, 0] = GT[:, 0]-T0
    return GT, Z
if __name__ == '__main__':
    X, Z = read_data()
