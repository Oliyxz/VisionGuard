import torch
import time
import pandas as pd
from tqdm import tqdm

def evaluate_clean_accuracy(model, dataloader, device='cuda'):
    """
    Evaluate baseline clean accuracy and inference latency.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating clean"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    latency = time.time() - start_time
    accuracy = 100 * correct / total
    
    return accuracy, latency

def evaluate_attack_success_rate(model, attack, dataloader, device='cuda'):
    """
    Evaluate model robustness against a specific attack.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Evaluating attack"):
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial images
        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    robust_accuracy = 100 * correct / total
    attack_success_rate = 100 - robust_accuracy
    
    return attack_success_rate
