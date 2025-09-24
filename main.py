from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    data_dir = "data"   # path to dataset folder
    train_model(data_dir, epochs=5, lr=0.001)
    evaluate_model(data_dir)