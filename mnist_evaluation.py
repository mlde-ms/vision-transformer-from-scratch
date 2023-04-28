import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import numpy as np

# from vit import VisionTransformer

def evaluate(model):
    # model = VisionTransformer(28, 2, 2, 2, 64, 256, num_classes=10, representation_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU for training
    print(f"Using {device} for evaluation.")
    model = model.to(device)  # Use GPU for training

    # Load the saved weights into the new model
    model.load_state_dict(torch.load('vit_weights.pth'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 256

    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model.eval()

    # Track the total number of correct predictions
    total_correct = 0

    # Initialize numpy arrays to store predicted and true labels for all batches
    all_predicted = np.zeros(len(test_data), dtype=int)
    all_targets = np.zeros(len(test_data), dtype=int)


    # Disable gradient computation to speed up the inference
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)  # Use GPU for training
            output = model(data)
            # Calculate the predicted class labels
            _, predicted = torch.max(output, dim=1)
            # Update total number of correct predictions
            total_correct += (predicted == target).sum().item()
            
            # Fill the numpy arrays with batch-wise predictions and targets
            start = batch_idx * batch_size
            end = start + len(predicted)
            all_predicted[start:end] = predicted.cpu().numpy()
            all_targets[start:end] = target.cpu().numpy()

    # calculate the accuracy
    accuracy = total_correct / len(test_data)
    print('Accuracy on the train data: {:.2%}'.format(accuracy))
    print(confusion_matrix(all_predicted, all_targets))
