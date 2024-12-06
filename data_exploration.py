import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


base_dir = Path('aml-2024-feather-in-focus')
class_names = np.load(base_dir / "class_names.npy", allow_pickle=True).item()
train_df = pd.read_csv(base_dir / 'train_images.csv')
test_df = pd.read_csv(base_dir / 'test_images_path.csv')
attributes = np.load(base_dir / 'attributes.npy', allow_pickle=True)

with open(base_dir / 'attributes.txt', 'r') as f:
    attributes_names = f.read().splitlines()

label_to_name = {v: k.split('.')[1] for k, v in class_names.items()}
train_df['bird_name'] = train_df['label'].map(label_to_name)

# class_distribution = train_df['label'].value_counts()
plt.figure(figsize=(15, 6))
ax = sns.countplot(data=train_df, y='bird_name')
plt.title('Distribution of Bird Species in Training Set')
plt.xlabel('Count')
plt.ylabel('Species Name')
plt.show()

plt.figure(figsize=(15, 6))
sns.histplot(data=train_df, x='label')
plt.title('Distribution of Classes')
plt.xlabel('Species Label')
plt.ylabel('Count')

sample_path = train_df['image_path'].iloc[0].lstrip('/') 
sample_image_path = base_dir / 'train_images' / sample_path
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path)
sample_image

# check image size and format
image_sizes = []
for path in train_df['image_path'][:100]:
    clean_path = path.lstrip('/')
    full_path = base_dir / 'train_images' / clean_path
    img = tf.keras.preprocessing.image.load_img(full_path)
    image_sizes.append(img.size)


