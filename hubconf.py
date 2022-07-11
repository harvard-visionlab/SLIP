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

def vits16_clip_25ep_yfcc15M(pretrained=True, **kwargs):
	"""
	CLIP_VITS16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.CLIP_VITS16(**kwargs)
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

def vits16_simclr_25ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SIMCLR_VITS16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SIMCLR_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/simclr_small_25ep.pt"
		cache_file_name = "simclr_small_25ep-11acf52e.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '11acf52e'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vits16_slip_25ep_yfcc15M(pretrained=True, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_small_25ep.pt"
		cache_file_name = "slip_small_25ep-0e78b02f.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '0e78b02f'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vits16_slip_50ep_yfcc15M(pretrained=True, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_small_50ep.pt"
		cache_file_name = "slip_small_50ep-b9a8fd80.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'b9a8fd80'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vits16_slip_100ep_yfcc15M(pretrained=True, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_small_100ep.pt"
		cache_file_name = "slip_small_100ep-cc896760.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'cc896760'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform
