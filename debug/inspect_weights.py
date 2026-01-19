
import torch
import sys
import yaml


weight_path = r"c:/Users/paubu/Desktop/Master/Setembre 2025 - Gener 2026/TFM/Code/SpikeYOLO_Encoded/69M_best.pt"
sys.path.append(r"c:/Users/paubu/Desktop/Master/Setembre 2025 - Gener 2026/TFM/Code/SpikeYOLO_Encoded")


try:
    print(f"Loading {weight_path}...")
    ckpt = torch.load(weight_path, map_location='cpu')
    
    print("\n--- Keys in Checkpoint ---")
    print(ckpt.keys())
    
    if 'model' in ckpt:
        model = ckpt['model']
        print("\n--- Model Class ---")
        print(type(model))
        
        print("\n--- Model Args (Config) ---")
        if hasattr(model, 'args'):
            print(model.args)
        elif hasattr(model, 'yaml'):
            print(model.yaml)
        else:
            print("No explicit args/yaml found on model object.")

        # Try to find MS_GetT layer to see its param
        print("\n--- MS_GetT Configuration in Model ---")
        found = False
        for name, module in model.named_modules():
            if 'MS_GetT' in str(type(module)):
                print(f"Found MS_GetT at {name}: {module}")
                if hasattr(module, 'T'):
                    print(f"  -> module.T = {module.T}")
                found = True
        if not found:
            print("No MS_GetT module found in the loaded model.")
            
    else:
        print("No 'model' key in checkpoint.")

except Exception as e:
    print(f"Error loading model: {e}")
