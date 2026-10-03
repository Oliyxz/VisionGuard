import os
import pandas as pd
import numpy as np

def generate_mock_results():
    """
    Generate realistic mock results for the dashboard to demonstrate functionality.
    Running real evaluations on ViT and ConvNeXt takes significant compute time,
    so this script scaffolds the output format.
    """
    models = ["ViT-B/16", "ConvNeXt-Tiny"]
    precisions = ["FP32", "FP16", "INT8"]
    attacks = ["FGSM", "PGD"]
    epsilons = [0, 2, 4, 8, 16]
    
    results = []
    
    for model in models:
        for precision in precisions:
            for attack in attacks:
                for eps in epsilons:
                    if eps == 0:
                        asr = 0.0
                        clean_acc = 82.5 if model == "ViT-B/16" else 79.2
                    else:
                        # Generate some realistic looking ASR curves
                        # ViT is generally slightly more robust than ConvNeXt
                        # INT8 is generally more vulnerable than FP32
                        base_vuln = 1.0 if model == "ConvNeXt-Tiny" else 0.85
                        prec_mult = 1.0 if precision == "FP32" else (1.1 if precision == "FP16" else 1.3)
                        attack_mult = 1.0 if attack == "FGSM" else 1.4
                        
                        asr = min(100.0, eps * 6.0 * base_vuln * prec_mult * attack_mult + np.random.normal(0, 2))
                        clean_acc = 82.5 if model == "ViT-B/16" else 79.2
                        
                    results.append({
                        "Model": model,
                        "Precision": precision,
                        "Attack": attack,
                        "Epsilon": eps,
                        "Clean Acc (%)": clean_acc,
                        "ASR (%)": max(0, asr)
                    })

    df = pd.DataFrame(results)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/results.csv', index=False)
    print("Successfully generated data/results.csv")

if __name__ == "__main__":
    generate_mock_results()
