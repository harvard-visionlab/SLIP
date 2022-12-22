'''
	hubconf for SLIP models
'''

import os
import torch
import torchvision

import models 

print(dir(models))

print(os.path.abspath(models.__file__))

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
	model = models.SIMCLR_VITS16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
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

def vits16_slip_25ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITS16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITS16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
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

def vits16_slip_50ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITS16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITS16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
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

def vits16_slip_100ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITS16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITS16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
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

# ===================================================================
#  ViT-big
# ===================================================================

def vitb16_clip_25ep_yfcc15M(pretrained=True, **kwargs):
	"""
	CLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.CLIP_VITB16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/clip_base_25ep.pt"
		cache_file_name = "clip_base_25ep-201382ca.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '201382ca'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_simclr_25ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SIMCLR_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SIMCLR_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/simclr_base_25ep.pt"
		cache_file_name = "simclr_base_25ep-c597f769.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'c597f769'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_slip_25ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_base_25ep.pt"
		cache_file_name = "slip_base_25ep-a37da619.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'a37da619'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_slip_50ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_base_50ep.pt"
		cache_file_name = "slip_base_50ep-1c346d5d.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '1c346d5d'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_slip_100ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt"
		cache_file_name = "slip_base_100ep-cdd23730.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'cdd23730'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

# ===================================================================
#  ViT-large
# ===================================================================

def vitl16_clip_25ep_yfcc15M(pretrained=True, **kwargs):
	"""
	CLIP_VITL16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.CLIP_VITL16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/clip_large_25ep.pt"
		cache_file_name = "clip_large_25ep-a8c4f7e3.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'a8c4f7e3'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitl16_simclr_25ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SIMCLR_VITL16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SIMCLR_VITL16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/simclr_large_25ep.pt"
		cache_file_name = "simclr_large_25ep-952bc376.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '952bc376'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitl16_slip_25ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITL16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITL16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_large_25ep.pt"
		cache_file_name = "slip_large_25ep-ca86c4de.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'ca86c4de'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitl16_slip_50ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITL16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITL16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_large_50ep.pt"
		cache_file_name = "slip_large_50ep-3300707a.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '3300707a'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitl16_slip_100ep_yfcc15M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITL16 (pre-trained on yfcc15M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITL16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_large_100ep.pt"
		cache_file_name = "slip_large_100ep-13c506e6.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '13c506e6'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

# ===================================================================
#  ViT-big (other datasets)
# ===================================================================

def vitb16_clip_40ep_cc3M(pretrained=True, **kwargs):
	"""
	CLIP_VITB16 (pre-trained on Conceptual Captions 3M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.CLIP_VITB16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/clip_base_cc3m_40ep.pt"
		cache_file_name = "clip_base_cc3m_40ep-f017d438.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'f017d438'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_clip_35ep_cc12M(pretrained=True, **kwargs):
	"""
	CLIP_VITB16 (pre-trained on Conceptual Captions 12M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.CLIP_VITB16(**kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/clip_base_cc12m_35ep.pt"
		cache_file_name = "clip_base_cc12m_35ep-44e869f1.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = '44e869f1'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_slip_40ep_cc3M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on Conceptual Captions 3M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_base_cc3m_40ep.pt"
		cache_file_name = "slip_base_cc3m_40ep-d3215d15.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'd3215d15'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform

def vitb16_slip_35ep_cc12M(pretrained=True, ssl_mlp_dim=4096, ssl_emb_dim=256, **kwargs):
	"""
	SLIP_VITB16 (pre-trained on Conceptual Captions 12M)
	pretrained (bool): kwargs, load pretrained weights into the model
	"""
	model = models.SLIP_VITB16(ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)
	if pretrained:
		checkpoint_url = "https://dl.fbaipublicfiles.com/slip/slip_base_cc12m_35ep.pt"
		cache_file_name = "slip_base_cc12m_35ep-d8a0150a.pt"
		checkpoint = torch.hub.load_state_dict_from_url(
			url=checkpoint_url, 
			map_location='cpu',
			file_name=cache_file_name,
			check_hash=True
		)
		state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
		model.load_state_dict(state_dict, strict=True)
		model.hashid = 'd8a0150a'
		model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", cache_file_name)
		
	transform = _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	return model, transform
