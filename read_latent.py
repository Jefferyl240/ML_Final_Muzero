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
for reward in reward_values[:10]:  # First 10 unique rewards
    idx = np.where(rewards == reward)[0][0]  # Get first occurrence of this reward
    img = images[idx]
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Image with reward: {reward}')
    plt.axis('off')
    plt.savefig(f'reward_{reward}_image.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved image for reward {reward}")

# Plot reward distribution with better binning
plt.figure(figsize=(12, 6))

# Create subplot for histogram
plt.subplot(1, 2, 1)
# Use unique reward values as bin edges to ensure each reward value gets its own bin
bin_edges = np.concatenate([reward_values - 0.5, [reward_values[-1] + 0.5]])
hist, bins, _ = plt.hist(rewards, bins=bin_edges, edgecolor='black')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution (Exact Values)')
plt.xticks(reward_values)  # Show all unique reward values on x-axis

# Create subplot for bar plot
plt.subplot(1, 2, 2)
plt.bar(reward_values, [np.sum(rewards == r) for r in reward_values], edgecolor='black')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution (Bar Plot)')
plt.xticks(reward_values)  # Show all unique reward values on x-axis

plt.tight_layout()
plt.savefig('reward_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Print detailed reward statistics
print("\nDetailed reward statistics:")
print(f"Total number of samples: {len(rewards)}")
print(f"Number of unique rewards: {len(reward_values)}")
print("\nReward value counts:")
for reward in reward_values:
    count = np.sum(rewards == reward)
    percentage = (count / len(rewards)) * 100
    print(f"Reward {reward}: {count} samples ({percentage:.1f}%)")

print("\nReward distribution saved as 'reward_distribution.png'")