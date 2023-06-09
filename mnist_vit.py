import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix


def train(model):
    # Use GPU for training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using {device} for training.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 256
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model.state_dict(), 'vit_weights.pth')


def evaluate(model):
    # Use GPU for evaluation if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using {device} for evaluation.")

    # Load the saved weights into the new model
    model.load_state_dict(torch.load('vit_weights.pth', map_location=torch.device(device)))

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


def visualize_attention(model):
    # Load the saved weights into the new model
    model.load_state_dict(torch.load('vit_weights.pth', map_location=torch.device('cpu')))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 10

    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model.eval()

    # Disable gradient computation to speed up the inference
    with torch.no_grad():
        # Get a batch of test data and iterate over it
        images, labels = next(iter(test_loader))
        for i in range(batch_size):

            image = images[i]
            label = labels[i].item()

            # Add a dummy batch dimension of 1
            image = image[None]
            # Get the model prediction
            _ = model(image)

            for layer in range(2):
                for head in range(2):
                    attention_output = model.get_attention_output(image, layer, head)
                    visualize_attention_head(attention_output[0], 196, 2, f'layer_{layer}_head_{head}.png', f'attention/{i}_{label}')

            original_image = images[i].squeeze().numpy()
            plt.imshow(original_image, cmap='gray')
            plt.axis('off')
            plt.savefig(f'attention/{i}_{label}/img.png', bbox_inches='tight')
            plt.close()


def visualize_attention_head(attention_head_output, num_patches, patch_size, output_file, output_dir):
    # Tensor shape: (1025, 128) -> (1024, 128)
    reshaped_output = attention_head_output[1:, :]
    # Take the mean of the 128 values for each patch (1024, 128) -> (1024)
    patch_values = torch.mean(reshaped_output, dim=-1).detach().numpy()
    # (1024) -> (32, 32)
    patch_dim = int(np.sqrt(num_patches))
    patch_values_2d = patch_values.reshape(patch_dim, patch_dim)
    # (32, 32) -> multiplied by patch_size -> (512, 512)
    scaled_patch_values = np.repeat(np.repeat(patch_values_2d, patch_size, axis=0), patch_size, axis=1)
    # Visualize the image in a heatmap
    plt.imshow(scaled_patch_values, cmap='viridis')
    plt.colorbar()
    # Save the figure to the specified file and directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_file), bbox_inches='tight', dpi=100)
    plt.clf()
