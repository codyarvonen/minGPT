import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":

    file_path = "loss.pkl"

    with open(file_path, 'rb') as file:
        loss = pickle.load(file)

    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
