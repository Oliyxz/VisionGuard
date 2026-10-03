import torch
import torch.quantization

def apply_static_quantisation_int8(model, dataloader_for_calibration, device='cpu'):
    """
    Apply Post-Training Static Quantisation (INT8) to the given PyTorch model.
    """
    model.eval()
    model.to('cpu') # Quantisation is generally supported better on CPU

    # Define quantisation configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for static quantisation
    model_prepared = torch.quantization.prepare(model)
    
    # Calibration step
    print("Calibrating model for quantisation...")
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader_for_calibration):
            model_prepared(images)
            if i >= 10: # Calibrate on a small subset
                break
                
    # Convert to quantised model
    model_quantised = torch.quantization.convert(model_prepared)
    
    return model_quantised

def apply_fp16_quantisation(model):
    """
    Convert model to FP16 precision.
    """
    return model.half()
