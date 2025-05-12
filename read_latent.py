#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.load('latent_dataset.npz')
latent_vectors = data['latent']
images = data['image']
rewards = data['reward']
ids = data['id']

# Print dataset info
print(f"Dataset shape: {latent_vectors.shape}")
print(f"Image shape: {images.shape}")
print(f"Number of samples: {len(rewards)}")

# Print image statistics
print("\nImage statistics:")
print(f"Min value: {images.min()}")
print(f"Max value: {images.max()}")
print(f"Mean value: {images.mean()}")

# Find the highest reward sample
max_reward_idx = np.argmax(rewards)
print(f"\nHighest reward sample:")
print(f"Reward: {rewards[max_reward_idx]}")
print(f"ID: {ids[max_reward_idx]}")

# Get the image
img = images[max_reward_idx]
print(f"\nImage stats for highest reward sample:")
print(f"Min value: {img.min()}")
print(f"Max value: {img.max()}")
print(f"Mean value: {img.mean()}")
print(f"Shape: {img.shape}")

# Display the image
plt.figure(figsize=(8, 8))
# The images should already be in [0, 1] range and [H, W, 3] format
plt.imshow(img)
plt.title(f'Image with highest reward: {rewards[max_reward_idx]}')
plt.axis('off')
plt.savefig('highest_reward_image.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nImage saved as 'highest_reward_image.png'")

# Also save a few more images with different rewards for comparison
reward_values = np.unique(rewards)
print(f"\nUnique reward values: {reward_values}")

# Save images for a few different reward values
for reward in reward_values[:10]:  # First 5 unique rewards
    idx = np.where(rewards == reward)[0][0]  # Get first occurrence of this reward
    img = images[idx]
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Image with reward: {reward}')
    plt.axis('off')
    plt.savefig(f'reward_{reward}_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved image for reward {reward}")

reward_distribution = np.histogram(rewards, bins=100, range=(0, 100))
plt.figure(figsize=(10, 6))
plt.bar(reward_distribution[1][:-1], reward_distribution[0], width=0.5)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution')
plt.savefig('reward_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Reward distribution saved as 'reward_distribution.png'")