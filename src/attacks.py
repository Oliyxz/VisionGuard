import torchattacks

def get_fgsm_attack(model, eps=8/255):
    """
    Configure Fast Gradient Sign Method (FGSM) attack.
    """
    attack = torchattacks.FGSM(model, eps=eps)
    return attack

def get_pgd_attack(model, eps=8/255, alpha=2/255, steps=10):
    """
    Configure Projected Gradient Descent (PGD) attack.
    """
    attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    return attack
