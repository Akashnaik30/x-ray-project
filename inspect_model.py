import torch
import torchvision.models as models
import os

model_path = os.path.join("model", "xray_model.pth")
try:
    model = models.densenet121()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Clean keys if they have 'module.' prefix (e.g. from DataParallel)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    print("Successfully loaded state_dict into torchvision.models.densenet121!")
except Exception as e:
    print(f"Error loading into densenet121: {e}")
