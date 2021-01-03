#!/bin/env python3
import torch
import numpy as np
from models import SubtitleSegmentation, BaselineModel
from skimage import io, transform

width, height = (640, 360)
input_width, input_height = (640, 360) # DR Live

if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = SubtitleSegmentation(in_channels=3, height=640, width=360).to(device)
	model.load_state_dict(torch.load("subseg-batch_size-8.num_epochs-600.torch", map_location=device))
	
	#model = BaselineModel(in_channels=3, height=640, width=360).to(device)
	#model.load_state_dict(torch.load("baseline-batch_size-8.num_epochs-600.torch", map_location=device))

	image = io.imread("data/sintel/frames/frame128.png")
	image = transform.resize(image, (height, width))
	image = image / image.max()
	image = image.transpose(2, 0, 1)
	image = image[None, ...]
	image = torch.tensor(image).float().to(device)

	mask, has_subtitle = model(image)
	mask = torch.sigmoid(mask)
	has_subtitle = torch.sigmoid(has_subtitle)

	has_subtitle = has_subtitle.item()

	print("Has subtitle!" if has_subtitle > 0.5 else "Does not contain any subtitles....")

	# Perform masking
	mask = (mask > 0.5).float()
	masked = mask * image

	# Convert CHW to CHW
	mask = mask.permute(0, 2, 3, 1) 
	masked = masked.permute(0, 2, 3, 1)
	
	mask = mask.detach().cpu().numpy()
	masked = masked.detach().cpu().numpy()

	mask = mask * 255
	masked = masked * 255

	mask = np.array(mask, np.uint8)
	masked = np.array(masked, np.uint8)

	io.imsave("mask.png", mask[0])
	io.imsave("masked.png", masked[0])

