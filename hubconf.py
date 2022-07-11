'''
	hubconf for SLIP models
'''

import os
import torch
import torchvision

import models 

dependencies = ['torch', 'torchvision']

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform

# ===================================================================
#  ViT-Small
# ===================================================================

def clip_vits16_yfcc15M_in1k(pretrained=True, **kwargs):
	"""
	CLIP_VITS16 (pre-trained on yfcc15M, fine-tuned on imagenet-1k)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.CLIP_VITS16()
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/clip_small_25ep.pt"
		cache_file_name = "clip_small_25ep-5d54c95a.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '5d54c95a'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform
