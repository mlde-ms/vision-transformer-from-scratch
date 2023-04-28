import matplotlib.pyplot as plt
import numpy as np
import torch
import os


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
    # plt.show()
    # Save the figure to the specified file and directory
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_file), bbox_inches='tight', dpi=100)
    plt.clf()
