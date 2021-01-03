import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

class SubtitleSegmentationDataset(Dataset):
	def __init__(self, root_dir, csv_filename="data.csv", transform=None):
		self.root_dir = root_dir
		self.csv = pd.read_csv(os.path.join(root_dir, csv_filename))
		self.transform = transform
			
	def __len__(self):
		return len(self.csv)
	
	def __getitem__(self, idx):
		image_path, mask_path, has_subtitle = self.csv.iloc[idx]
		image_path = os.path.join(self.root_dir, image_path)
		mask_path = os.path.join(self.root_dir, mask_path)
		
		image = io.imread(image_path)
		mask = io.imread(mask_path)[..., np.newaxis]

		image = image / 255
		mask = mask / 255
		
		sample = { "image": image, "mask": mask, "has_subtitle": has_subtitle }
		
		if self.transform:
				sample = self.transform(sample)
		
		return sample