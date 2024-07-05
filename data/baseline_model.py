import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml
from dataset_loader import CustomDataset, custom_standardize_transform, custom_pscaling_transform
from torchvision import transforms
from tqdm import tqdm

class DoubleConv(nn.Module):
    '''
        Two Convolutional layers with BatchNorm2d and ReLU after each.
        
        Attributes:
        ----------
        in_ch: int
            the number of input channels
        out_ch: int
            the number of output channels
    '''
    def __init__(self, in_ch : int, out_ch : int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    '''
        The UNet model.
        
        Attributes:
        ----------
        in_ch: int
            the number of input channels
        out_ch: int
            the number of output channels
    '''
    def __init__(self, in_ch : int = 7, out_ch : int = 7):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d3 = self.dec3(torch.cat([self.upsample(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))

        return self.out_conv(d1)
    

def train_model(model : torch.nn.Module, train_loader : DataLoader, num_epochs : int = 10, learning_rate : float = 0.001, weight_decay : float = 0.0001, device : str = 'cpu', model_path : str = "model.pth") :
    '''
    Trains the given model using the dataloader provided and saves its best state.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        The DataLoader for the training data.
    test_loader : torch.utils.data.DataLoader
        The DataLoader for evaluating the model.
    num_epochs : int, optional
        The number of epochs to train the model (default is 10).
    learning_rate : float, optional
        The learning rate for the optimizer (default is 0.001).
    weight_decay : float, optional
        The weight decay (L2 penalty) for the optimizer (default is 0.0001).
    device : str, optional
        The device to use for training, either 'cpu' or 'cuda' (default is 'cpu').
    model_path : str, optional
        The path to save the best model state (default is "model.pth").

    Returns:
    -------
    None
    
    '''
    
    model.to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_train_loss = float('inf')

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training:")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        for i, (noisy_images, images, _) in progress_bar:
            noisy_images = noisy_images.to(device)
            images = images.to(device)
                            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            progress_bar.set_description(f"Train Loss: {loss.item():.4f}")
            
        train_loss /= len(train_loader.dataset)
        
        print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}' 
               .format(epoch+1, num_epochs, i+1, total_step, train_loss))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch}, with train loss: {train_loss}')

def evaluate_model(model : torch.nn.Module, test_loader : DataLoader, plot_channels: bool = False, loss_cutoff : float = 0.01, device : str = 'cpu'):
    '''
    Evaluates the given model using the test data.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    test_loader : torch.utils.data.DataLoader
        The DataLoader for the test data.
    loss_cutoff : float, optional
        The loss cutoff to determine if an image is normal or abnormal (default is 0.01).
        
    Returns:
    -------
    None
    
    '''
    print('Evaluating model...')
    model.eval()
    model.to(device)
    
    losses = {0: [], 1: [], 2: []}
    counts = {0: 0, 1: 0, 2: 0}
    
    normal_labels = []
    abnormal_labels = []
    
    for _, images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        for i in range(len(images)):
            loss = torch.nn.functional.mse_loss(outputs[i], images[i]).item()
            label = labels[i].item()
            losses[label].append(loss)
            counts[label] += 1

            if plot_channels and counts[label] < 8:
                channels = ['dem', 'red', 'green', 'blue']
                classes = ['Case 1', 'Case 2', 'Normal']
                fig, axs = plt.subplots(2, 4, figsize=(16, 8))
                for k in range(4):
                    # plot input channels
                    axs[0][k].imshow(images[i].cpu().detach().numpy()[k], cmap='gray')
                    axs[0][k].set_title(f"Input channel: {channels[k]}")
                    axs[0][k].axis('off')
                    # plot output channels
                    axs[1][k].imshow(outputs[i].cpu().detach().numpy()[k], cmap='gray')
                    axs[1][k].set_title(f"Output channel: {channels[k]}")
                    axs[1][k].axis('off')
                plt.suptitle(f"Loss: {loss:.4f}, Label: {classes[label]}", size=20)
                plt.show()
            elif not plot_channels:
                plt.scatter(loss, label)
    
    normal_labels = sum(1 for loss in losses[2] if loss < loss_cutoff)
    abnormal_labels = {0: sum(1 for loss in losses[0] if loss >= loss_cutoff),1: sum(1 for loss in losses[1] if loss >= loss_cutoff)}
    
    print(f'Normal acc: {normal_labels/counts[2]:.4f} - ({normal_labels}/{counts[2]})')
    print(f'Case 1 acc: {abnormal_labels[0]/counts[0]:.4f} - ({abnormal_labels[0]}/{counts[0]})')
    print(f'Case 2 acc: {abnormal_labels[1]/counts[1]:.4f} - ({abnormal_labels[1]}/{counts[1]})')
    
    total_correct = normal_labels + abnormal_labels[0] + abnormal_labels[1]
    total_count = sum(counts.values())
    print(f'Overall acc: {total_correct/total_count:.4f}')
    
    avg_losses = {label: np.mean(losses[label]) for label in losses}
    print(f'Normal loss: {avg_losses[2]:.4f}, Case 1 loss: {avg_losses[0]:.4f}, Case 2 loss: {avg_losses[1]:.4f}')
    
    plt.show()

def optimise_threshold(model : torch.nn.Module, test_loader : DataLoader, device : str = 'cpu'):
    '''
    Optimizes the threshold for the model using the test data.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    test_loader : torch.utils.data.DataLoader
        The DataLoader for the test data.
        
    Returns:
    -------
    None
    
    '''
    print('Optimizing threshold...')
    model.eval()
    model.to(device)
    normal_loss = []
    case_1_loss = []
    case_2_loss = []
    
    for _, images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        for i in range(len(images)):
            loss = torch.nn.functional.mse_loss(outputs[i], images[i])
            if labels[i] == 2:
                normal_loss.append(loss.item())
            elif labels[i] == 0:
                case_1_loss.append(loss.item())
            elif labels[i] == 1:
                case_2_loss.append(loss.item())
    
    normal_loss = np.array(normal_loss)
    case_1_loss = np.array(case_1_loss)
    case_2_loss = np.array(case_2_loss)
    
    thresholds = normal_loss
    best_acc = (0, 0, 0, 0)
    best_threshold = 0
    for threshold in thresholds:
        normal_acc = (normal_loss < threshold).sum() / len(normal_loss)
        case_1_acc = (case_1_loss > threshold).sum() / len(case_1_loss)
        case_2_acc = (case_2_loss > threshold).sum() / len(case_2_loss)
        overall_acc = ((normal_loss < threshold).sum() + (case_1_loss > threshold).sum() + (case_2_loss > threshold).sum()) / (len(normal_loss) + len(case_1_loss) + len(case_2_loss))
        if overall_acc > best_acc[0]:
            best_acc = (overall_acc, case_1_acc, case_2_acc, normal_acc)
            best_threshold = threshold
    
    print(f'Best threshold: {best_threshold:.4f}, Best acc: {best_acc[0]:.4f}, Case 1 acc: {best_acc[1]:.4f}, Case 2 acc: {best_acc[2]:.4f}, Normal acc: {best_acc[3]:.4f}')


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Hyper-parameters
    try:
        with open("baseline_config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        num_epochs = cfg["num_epochs"]
        batch_size = cfg["batch_size"]
        learning_rate = cfg["learning_rate"]
        weight_decay = cfg["weight_decay"]
        data_path = cfg["data_path"]
        data_stats_path = cfg["data_stats_path"]
        model_path = cfg["model_path"]
        noise_factor = cfg["noise_factor"]
        p = cfg["p"]
        resize_x = cfg["resize_x"]
        resize_y = cfg["resize_y"]
        channels = cfg["channels"]
        print(f'Config loaded: Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}, Weight decay: {weight_decay}, Noise factor: {noise_factor}, p: {p}, Resize : ({resize_x}, {resize_y})')
    except Exception as e:
        print("Error reading config file: \n", e)
        exit()

    
    # load data_stats from data_stats_path
    mean_path = os.path.join(data_stats_path, "train_means.npy")
    sdv_path = os.path.join(data_stats_path, "train_sdv.npy")
    
    transform = custom_standardize_transform(np.load(mean_path), np.load(sdv_path), device)

    percentiles_path = os.path.join(data_stats_path, "train_percentiles.npy")
    #transform = custom_pscaling_transform(np.load(percentiles_path))

    # Initialize model
    model = UNet(channels, channels)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from", model_path)
        test_data = CustomDataset(data_path + "val", transform, (resize_x, resize_y))
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        evaluate_model(model, test_loader, plot_channels=True,loss_cutoff=0.018,device=device)
        optimise_threshold(model, test_loader, device) 
    else:
        print(f'No saved model found at {model_path}. Training new model...')
        # Load data
        train_data = CustomDataset(data_path + "train", transform, (resize_x, resize_y), noise_factor=noise_factor, p=p)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_model(model, train_loader, num_epochs, learning_rate, weight_decay, device, model_path)