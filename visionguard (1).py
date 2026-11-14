"""
VisionGuard experiment library
==============================
Evaluating adversarial robustness and quantisation trade-offs of ViT-S/16 and
ConvNeXt-T under FP32 / FP16 / INT8 (static PTQ), FGSM and PGD (l_inf).

Pipeline per (dataset, architecture) pair
    1. fine-tune FP32 model                       -> Gate 1
    2. calibrate + build INT8 model and QSim proxy -> Gate 2
    3. attack grid (scenarios A0 / A1 / A2)        -> per-sample predictions (npz)
    4. gradient-masking diagnostics (D4, D5), K=50 check
    5. efficiency (size, compression ratio, latency)
    6. analysis (ASR + bootstrap CI, McNemar + Holm, tipping points, D1-D3, plots)

Design notes
    * Quantisation uses PyTorch FX graph-mode with a fake-quantised model that is
      calibrated once. The INT8 model is produced with convert_fx from that very
      same calibrated model, so QSim and INT8 share scales/zero-points exactly.
    * Only the backbone is quantised. Resize (bilinear, to 224) and ImageNet
      normalisation are float layers at the front of every model, so attacks
      work on native-resolution pixels in [0, 1].
    * Attention is un-fused (TIMM_FUSED_ATTN=0) in ALL precisions so the graph is
      identical everywhere and traceable; attention matmuls/softmax stay float.
    * Operators kept in float (Config.float_ops): residual adds, class-token concatenation,
      layer-scale multiplications and depthwise convolutions; LayerNorm modules are excluded
      from quantisation by type. Which integer kernels actually run is recorded per model
      in gate2.json (profiler).
    * INT8 runs on the oneDNN backend; the default x86/fbgemm path collapsed ConvNeXt INT8.
    * FP16 white-box attacks use loss scaling (FP16_LOSS_SCALE), because unscaled
      half-precision input gradients underflow to zero for most pixels.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import platform
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TIMM_FUSED_ATTN", "0")  # must be set before timm is imported
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (must precede CUDA init)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

try:  # belt and braces: also disable at runtime
    timm.layers.set_fused_attn(False)
except Exception:  # pragma: no cover
    pass

warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao")
warnings.filterwarnings("ignore", message=".*deprecat.*")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass
class Config:
    # paths
    out_dir: str = "/content/drive/MyDrive/VisionGuard"
    data_dir: str = "/content/data"
    # experiment grid
    datasets: Tuple[str, ...] = ("cifar100", "tinyimagenet")
    archs: Tuple[str, ...] = ("vit", "convnext")
    arch_names: Dict[str, str] = field(
        default_factory=lambda: {
            "vit": "vit_small_patch16_224.augreg_in21k_ft_in1k",
            "convnext": "convnext_tiny.fb_in22k_ft_in1k",
        }
    )
    pretrained: bool = True
    model_input_size: int = 224
    seed: int = 0
    # fine-tuning (Table 2 of the paper)
    epochs: int = 15
    batch_size: int = 128
    lr: float = 5e-5
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    crop_pad: int = 4
    amp: bool = True
    gate1_acc: float = 0.80  # CIFAR-100 only
    # subsets
    eval_per_class: Dict[str, int] = field(default_factory=lambda: {"cifar100": 10, "tinyimagenet": 5, "synthetic": 10})
    calib_n: int = 512
    # quantisation / Gate 2
    gate2_acc_tol: float = 0.03
    gate2_agree: float = 0.95  # relaxed from 0.99; disclosed in the paper
    cpu_threads: int = 0  # 0 -> all cores
    int8_bs: int = 50
    # attacks
    eps_255: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    diag_eps_255: int = 64
    pgd_steps: int = 20
    pgd_alpha_mult: float = 2.5
    attack_bs: int = 50  # lower this if you hit CUDA out-of-memory (ConvNeXt-T @224)
    # diagnostics
    d5_n: int = 500
    d5_eps_255: Tuple[int, ...] = (2, 4, 8)
    temp_T: float = 10.0
    grad_tol: float = 1e-10
    k50_n: int = 200
    k50_eps_255: Tuple[int, ...] = (4, 8)
    # statistics
    n_boot: int = 1000
    taus: Tuple[float, ...] = (0.25, 0.5, 0.75)
    # latency
    warmup: int = 50
    iters: int = 200
    lat_batches: Tuple[int, ...] = (1, 32)
    # multi-seed study
    seeds: Tuple[int, ...] = (0, 1, 2)
    seed_eps_255: Tuple[int, ...] = (4, 8)
    # gates raise errors when failed (set False to continue regardless, but report it)
    strict_gates: bool = True
    # operation types left in FLOAT (not quantised). Residual adds carry large outliers that 8 bits cannot represent;
    # concatenation (class token + patches) mixes different scales. Same policy for every model/run.
    float_ops: Tuple[str, ...] = ("add", "cat", "layer_norm", "mul", "depthwise_conv")
    # explicit (dataset, arch) pairs to run; None -> every dataset x every architecture.
    # Lets two people split the grid evenly, e.g. (("cifar100","convnext"), ("tinyimagenet","vit")).
    pairs: Optional[Tuple[Tuple[str, str], ...]] = None

    def all_pairs(self) -> List[Tuple[str, str]]:
        if self.pairs is not None:
            return [tuple(p) for p in self.pairs]
        return [(d, a) for d in self.datasets for a in self.archs]

    def run_dir(self, ds: str, arch: str) -> str:
        d = os.path.join(self.out_dir, f"{ds}_{arch}")
        os.makedirs(d, exist_ok=True)
        return d


class GateError(RuntimeError):
    pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_precision_flags(deterministic: bool = True):
    """True FP32 for evaluation/attacks: no TF32; deterministic algorithms where supported."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def set_cpu(cfg: Config):
    n = cfg.cpu_threads or (os.cpu_count() or 1)
    torch.set_num_threads(n)
    engines = torch.backends.quantized.supported_engines
    # oneDNN: the default x86/fbgemm path collapsed ConvNeXt INT8 on Colab CPUs (integer saturation); oneDNN matched QSim
    torch.backends.quantized.engine = "onednn" if "onednn" in engines else ("x86" if "x86" in engines else "fbgemm")
    return n


def env_info() -> dict:
    info = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        timm=timm.__version__,
        platform=platform.platform(),
        cpu=platform.processor(),
        cpu_count=os.cpu_count(),
        quant_engine=torch.backends.quantized.engine,
    )
    try:
        import torchattacks

        info["torchattacks"] = torchattacks.__version__
    except Exception:
        pass
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


class ArrayData:
    """In-memory uint8 images (N,3,H,W) + int64 labels."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, num_classes: int):
        assert x.dtype == torch.uint8 and x.ndim == 4
        self.x, self.y, self.num_classes = x, y.long(), num_classes

    def __len__(self):
        return len(self.y)

    def floats(self, idx) -> torch.Tensor:
        return self.x[idx].float() / 255.0


def _cifar100(root):
    import torchvision

    out = []
    for train in (True, False):
        d = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        x = torch.from_numpy(np.asarray(d.data)).permute(0, 3, 1, 2).contiguous()
        out.append(ArrayData(x, torch.tensor(d.targets), 100))
    return out


def _tiny_from_hf(root):
    from datasets import load_dataset

    ds = load_dataset("zh-plus/tiny-imagenet", cache_dir=os.path.join(root, "hf"))
    res = []
    for split in ("train", "valid"):
        s = ds[split]
        xs = np.stack([np.asarray(im.convert("RGB")) for im in s["image"]])
        res.append(ArrayData(torch.from_numpy(xs).permute(0, 3, 1, 2).contiguous(), torch.tensor(s["label"]), 200))
    return res


def _tiny_from_zip(root):
    """Fallback: official zip. Val labels come from val_annotations.txt."""
    import urllib.request, zipfile
    from PIL import Image

    z = os.path.join(root, "tiny-imagenet-200.zip")
    base = os.path.join(root, "tiny-imagenet-200")
    if not os.path.isdir(base):
        if not os.path.exists(z):
            urllib.request.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", z)
        zipfile.ZipFile(z).extractall(root)
    wnids = [l.strip() for l in open(os.path.join(base, "wnids.txt"))]
    cls = {w: i for i, w in enumerate(wnids)}
    xs, ys = [], []
    for w in wnids:
        for f in sorted(glob.glob(os.path.join(base, "train", w, "images", "*.JPEG"))):
            xs.append(np.asarray(Image.open(f).convert("RGB")))
            ys.append(cls[w])
    train = ArrayData(torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous(), torch.tensor(ys), 200)
    xs, ys = [], []
    for l in open(os.path.join(base, "val", "val_annotations.txt")):
        f, w = l.split("\t")[:2]
        xs.append(np.asarray(Image.open(os.path.join(base, "val", "images", f)).convert("RGB")))
        ys.append(cls[w])
    val = ArrayData(torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous(), torch.tensor(ys), 200)
    return train, val


def _tiny(root):
    cache = os.path.join(root, "tinyimagenet_uint8.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        mk = lambda p: ArrayData(torch.from_numpy(z[p + "x"]), torch.from_numpy(z[p + "y"]), 200)
        return mk("tr_"), mk("te_")
    try:
        tr, te = _tiny_from_hf(root)
    except Exception as e:  # network/HF problems -> official zip
        print("HF download failed (", repr(e)[:120], ") -> falling back to official zip")
        tr, te = _tiny_from_zip(root)
    np.savez(cache, tr_x=tr.x.numpy(), tr_y=tr.y.numpy(), te_x=te.x.numpy(), te_y=te.y.numpy())
    return tr, te


def make_synthetic(n_train=1500, n_test=400, classes=10, size=32, seed=0):
    """Tiny learnable dataset used only for smoke tests (class = colour/stripe pattern)."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.rand(classes, 3, 1, 1, generator=g)
    stripes = (torch.arange(size) // 4 % 2).float()

    def gen(n):
        y = torch.arange(n) % classes
        base = protos[y] * (0.6 + 0.4 * stripes[None, None, None, :])
        x = (base + 0.15 * torch.rand(n, 3, size, size, generator=g)).clamp(0, 1)
        return ArrayData((x * 255).round().to(torch.uint8), y, classes)

    return gen(n_train), gen(n_test)


def load_dataset_arrays(cfg: Config, name: str) -> Tuple[ArrayData, ArrayData]:
    os.makedirs(cfg.data_dir, exist_ok=True)
    if name == "cifar100":
        return _cifar100(cfg.data_dir)
    if name == "tinyimagenet":
        return _tiny(cfg.data_dir)
    if name == "synthetic":
        return make_synthetic()
    raise ValueError(name)


def iterate(data: ArrayData, bs: int, shuffle: bool, device, pad: int = 0, gen: Optional[torch.Generator] = None):
    """Yield (float images in [0,1] on device, labels). Optional random-crop(pad)+h-flip on GPU."""
    n = len(data)
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        idx = order[i : i + bs]
        x = data.x[idx].to(device, non_blocking=True).float() / 255.0
        y = data.y[idx].to(device)
        if pad > 0:
            B, C, H, W = x.shape
            xp = F.pad(x, (pad, pad, pad, pad))
            oy = torch.randint(0, 2 * pad + 1, (B,), device=device)
            ox = torch.randint(0, 2 * pad + 1, (B,), device=device)
            rows = (oy[:, None] + torch.arange(H, device=device))[:, None, :, None]
            cols = (ox[:, None] + torch.arange(W, device=device))[:, None, None, :]
            x = xp[
                torch.arange(B, device=device)[:, None, None, None],
                torch.arange(C, device=device)[None, :, None, None],
                rows,
                cols,
            ]
            flip = torch.rand(B, device=device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        yield x, y


def stratified_subset(labels: torch.Tensor, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = labels.numpy()
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        idx.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False).tolist())
    return np.sort(np.array(idx))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


class Front(nn.Module):
    """Bilinear up-sampling + ImageNet mean/std normalisation (float, differentiable)."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class Net(nn.Module):
    def __init__(self, front: nn.Module, backbone: nn.Module):
        super().__init__()
        self.front, self.backbone = front, backbone

    def forward(self, x):
        return self.backbone(self.front(x))


FP16_LOSS_SCALE = 2.0**14  # multiplies the logit gradient in FP16 attacks (sign-only attacks are unaffected by scaling)


class HalfNet(nn.Module):
    """FP16 wrapper: float32 in/out, half-precision compute (gradients flow in FP16).

    Gradients are scaled by FP16_LOSS_SCALE at the logits and non-finite input-gradient entries are
    zeroed, otherwise most pixel gradients underflow to exactly zero and PGD/FGSM silently fail.
    """

    def __init__(self, net: Net):
        super().__init__()
        self.net = copy.deepcopy(net).half()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
        out = self.net(x.half()).float()
        if out.requires_grad:
            out.register_hook(lambda g: g * FP16_LOSS_SCALE)
        return out


def build_net(cfg: Config, arch: str, num_classes: int, pretrained: Optional[bool] = None) -> Net:
    pretrained = cfg.pretrained if pretrained is None else pretrained
    kw = {}
    if arch == "vit" and cfg.model_input_size != 224:
        kw["img_size"] = cfg.model_input_size
    bb = timm.create_model(cfg.arch_names[arch], pretrained=pretrained, num_classes=num_classes, **kw)
    return Net(Front(cfg.model_input_size), bb)


# ----------------------------------------------------------------------------
# Fine-tuning (+ Gate 1)
# ----------------------------------------------------------------------------


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, bs: int, device, logits: bool = False):
    """x: CPU float tensor (N,3,H,W) in [0,1]. Returns numpy predictions (or logits)."""
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        o = model(x[i : i + bs].to(device)).float().cpu()
        out.append(o)
    o = torch.cat(out)
    return o.numpy() if logits else o.argmax(1).numpy()


@torch.no_grad()
def accuracy_on(model, data: ArrayData, device, bs=250) -> float:
    model.eval()
    correct = 0
    for x, y in iterate(data, bs, False, device):
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(data)


def finetune(cfg: Config, ds: str, arch: str, train: ArrayData, test: ArrayData, device) -> str:
    rd = cfg.run_dir(ds, arch)
    final = os.path.join(rd, "fp32_final.pt")
    if os.path.exists(final):
        print(f"[{ds}/{arch}] fine-tuned weights found -> skipping fine-tuning")
        return final
    set_seed(cfg.seed)
    net = build_net(cfg, arch, train.num_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total, warm = cfg.epochs * steps_per_epoch, max(1, int(cfg.warmup_frac * cfg.epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ck = os.path.join(rd, "ft_checkpoint.pt")
    start, hist = 0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=device)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        start, hist = s["epoch"], s["hist"]
        print(f"[{ds}/{arch}] resumed from epoch {start}")
    gen = torch.Generator().manual_seed(cfg.seed)
    for ep in range(start, cfg.epochs):
        gen.manual_seed(cfg.seed + ep)
        net.train()
        t0 = time.time()
        tl, nb = 0.0, 0
        for x, y in iterate(train, cfg.batch_size, True, device, pad=cfg.crop_pad, gen=gen):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(net(x), y)  # no label smoothing
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
            nb += 1
        acc = accuracy_on(net, test, device)
        hist.append(dict(epoch=ep + 1, train_loss=tl / nb, test_acc=acc, secs=time.time() - t0))
        print(
            f"[{ds}/{arch}] epoch {ep + 1}/{cfg.epochs} loss {tl / nb:.4f} test_acc {acc:.4f} ({time.time() - t0:.0f}s)"
        )
        torch.save(
            dict(
                net=net.state_dict(),
                opt=opt.state_dict(),
                sched=sched.state_dict(),
                scaler=scaler.state_dict(),
                epoch=ep + 1,
                hist=hist,
            ),
            ck,
        )
    torch.save(net.state_dict(), final)  # final epoch, no checkpoint selection on the test set
    pd.DataFrame(hist).to_csv(os.path.join(rd, "finetune_history.csv"), index=False)
    return final


def gate1(cfg: Config, ds: str, arch: str, net: Net, test: ArrayData, device) -> dict:
    acc = accuracy_on(net.to(device), test, device)
    passed = (acc > cfg.gate1_acc) if ds == "cifar100" else True  # Tiny-ImageNet: report only
    res = dict(
        dataset=ds,
        arch=arch,
        fp32_test_acc=acc,
        threshold=cfg.gate1_acc if ds == "cifar100" else None,
        passed=bool(passed),
    )
    json.dump(res, open(os.path.join(cfg.run_dir(ds, arch), "gate1.json"), "w"), indent=2)
    print(f"[{ds}/{arch}] GATE 1: FP32 test acc {acc:.4f} -> {'PASS' if passed else 'FAIL'}")
    if not passed and cfg.strict_gates:
        raise GateError(f"Gate 1 failed ({acc:.3f} <= {cfg.gate1_acc}). Train longer / tune lr before quantising.")
    return res


# ----------------------------------------------------------------------------
# Quantisation: INT8 (true integer kernels) + QSim fake-quant proxy
# ----------------------------------------------------------------------------

from torch.ao.quantization import (  # noqa: E402
    FakeQuantize,
    HistogramObserver,
    PerChannelMinMaxObserver,
    QConfig,
    QConfigMapping,
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx  # noqa: E402


def _qconfig_mapping(cfg: Config, model: Optional[nn.Module] = None) -> QConfigMapping:
    act = FakeQuantize.with_args(
        observer=HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    wt = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )
    import operator

    types = {
        "add": (operator.add, operator.iadd, torch.add, torch.Tensor.add),
        "mul": (operator.mul, torch.mul),
        "cat": (torch.cat,),
        "layer_norm": (nn.LayerNorm, F.layer_norm),
        "gelu": (nn.GELU, F.gelu),
        "softmax": (nn.Softmax, F.softmax),
    }
    m = QConfigMapping().set_global(QConfig(activation=act, weight=wt))
    for name in cfg.float_ops:
        for t in types.get(name, ()):
            m = m.set_object_type(t, None)  # None = leave this op type in float
    if (
        model is not None
    ):  # match by isinstance: timm's LayerNorm / LayerNorm2d are SUBCLASSES that set_object_type(nn.LayerNorm) misses
        for nm, mod in model.named_modules():
            if ("layer_norm" in cfg.float_ops and isinstance(mod, nn.LayerNorm)) or isinstance(mod, nn.Identity):
                m = m.set_module_name(
                    nm, None
                )  # Identity (drop_path) otherwise gets an observer that quantises the residual stream
    if (
        "depthwise_conv" in cfg.float_ops and model is not None
    ):  # depthwise convs read the residual stream directly (ConvNeXt)
        for nm, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.groups == mod.in_channels and mod.in_channels > 1:
                m = m.set_module_name(nm, None)
    return m


def _set_quant_mode(m: nn.Module, observe: bool):
    m.apply(enable_observer if observe else disable_observer)
    m.apply(disable_fake_quant if observe else enable_fake_quant)


def build_quantised(cfg: Config, net: Net, calib_x: torch.Tensor, device, state_path: Optional[str] = None):
    """Returns (qsim: fake-quant float model on `device`, int8: integer model on CPU).
    Per-channel symmetric INT8 weights, per-tensor asymmetric INT8 activations, histogram observers."""
    S = cfg.model_input_size
    bb = copy.deepcopy(net.backbone).cpu().train()
    prepared = prepare_qat_fx(bb, _qconfig_mapping(cfg, bb), (torch.randn(2, 3, S, S),))
    front = copy.deepcopy(net.front).to(device).eval()
    if state_path and os.path.exists(state_path):
        prepared.load_state_dict(torch.load(state_path, map_location="cpu"))
        print("  loaded calibrated quantisation state (scales/zero-points) from", state_path)
        prepared.eval()
    else:
        prepared.to(device).eval()
        _set_quant_mode(prepared, observe=True)
        with torch.no_grad():
            for i in range(0, len(calib_x), 64):
                prepared(front(calib_x[i : i + 64].to(device)))
        prepared.cpu()
        if state_path:
            torch.save(prepared.state_dict(), state_path)
    _set_quant_mode(prepared, observe=False)
    qsim = Net(front, copy.deepcopy(prepared).to(device).eval())
    int8 = Net(copy.deepcopy(net.front).cpu().eval(), convert_fx(copy.deepcopy(prepared).cpu().eval()))
    for p in qsim.parameters():
        p.requires_grad_(True)
    return qsim, int8.eval()


def integer_kernel_evidence(int8: nn.Module, x: torch.Tensor) -> List[str]:
    """Profiler evidence that integer kernels (quantized::*) actually execute."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            int8(x[:2].cpu())
    return sorted({e.key for e in prof.key_averages() if e.key.startswith("quantized::")})


def model_size_mb(model: nn.Module) -> float:
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


# ----------------------------------------------------------------------------
# Context for one (dataset, architecture) pair
# ----------------------------------------------------------------------------


@dataclass
class Ctx:
    cfg: Config
    ds: str
    arch: str
    device: torch.device
    train: ArrayData
    test: ArrayData
    fp32: Net
    fp16: Optional[nn.Module]
    qsim: Optional[Net] = None
    int8: Optional[Net] = None
    eval_idx: Optional[np.ndarray] = None
    x: Optional[torch.Tensor] = None  # eval images (CPU, float [0,1])
    y: Optional[torch.Tensor] = None
    clean: Optional[dict] = None

    @property
    def rd(self):
        return self.cfg.run_dir(self.ds, self.arch)


def setup_context(
    cfg: Config,
    ds: str,
    arch: str,
    data: Optional[Tuple[ArrayData, ArrayData]] = None,
    calib_seed: Optional[int] = None,
    quantise: bool = True,
    state_suffix: str = "",
) -> Ctx:
    device = get_device()
    set_cpu(cfg)
    set_precision_flags(deterministic=False)
    train, test = data if data is not None else load_dataset_arrays(cfg, ds)
    rd = cfg.run_dir(ds, arch)
    # 1) fine-tune + gate 1
    final = finetune(cfg, ds, arch, train, test, device)
    set_precision_flags(deterministic=True)  # evaluation / attacks: deterministic algorithms where supported
    fp32 = build_net(cfg, arch, train.num_classes, pretrained=False)
    fp32.load_state_dict(torch.load(final, map_location="cpu"))
    fp32 = fp32.to(device).eval()
    for p in fp32.parameters():
        p.requires_grad_(False)
    if not os.path.exists(os.path.join(rd, "gate1.json")):
        gate1(cfg, ds, arch, fp32, test, device)
    fp16 = HalfNet(fp32).eval() if device.type == "cuda" else None
    # 2) evaluation subset (class-stratified, fixed seed) and calibration set (disjoint, from train)
    idx = stratified_subset(test.y, cfg.eval_per_class[ds], seed=0)
    x, y = test.floats(torch.from_numpy(idx)), test.y[idx]
    ctx = Ctx(cfg, ds, arch, device, train, test, fp32, fp16, eval_idx=idx, x=x, y=y)
    if quantise:
        cs = cfg.seed if calib_seed is None else calib_seed
        rng = np.random.default_rng(1000 + cs)
        cidx = rng.choice(len(train), size=min(cfg.calib_n, len(train)), replace=False)
        calib = train.floats(torch.from_numpy(cidx))
        ctx.qsim, ctx.int8 = build_quantised(
            cfg, fp32, calib, device, state_path=os.path.join(rd, f"quant_state{state_suffix}.pt")
        )
        for m in (ctx.qsim, ctx.int8):
            m.eval()
    return ctx


def clean_predictions(ctx: Ctx) -> dict:
    cfg, dev = ctx.cfg, ctx.device
    out = dict(idx=ctx.eval_idx, y=ctx.y.numpy(), fp32=predict(ctx.fp32, ctx.x, 250, dev))
    out["fp16"] = predict(ctx.fp16, ctx.x, 250, dev) if ctx.fp16 is not None else None
    if ctx.qsim is not None:
        out["qsim"] = predict(ctx.qsim, ctx.x, 250, dev)
        out["int8"] = predict(ctx.int8, ctx.x, cfg.int8_bs, torch.device("cpu"))
    ctx.clean = out
    return out


def gate2(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    c = ctx.clean or clean_predictions(ctx)
    y = c["y"]
    acc = {k: float((c[k] == y).mean()) for k in ("fp32", "fp16", "qsim", "int8") if c.get(k) is not None}
    agree = float((c["qsim"] == c["int8"]).mean())
    ops = integer_kernel_evidence(ctx.int8, ctx.x)
    cond1 = (acc["fp32"] - acc["int8"]) <= cfg.gate2_acc_tol
    cond2 = agree >= cfg.gate2_agree
    cond3 = any(o in ops for o in ("quantized::linear", "quantized::conv2d", "quantized::linear_dynamic"))
    res = dict(
        dataset=ctx.ds,
        arch=ctx.arch,
        clean_acc=acc,
        acc_drop_fp32_int8=acc["fp32"] - acc["int8"],
        qsim_int8_agreement=agree,
        integer_ops=ops,
        cond_acc=bool(cond1),
        cond_agree=bool(cond2),
        cond_int_kernels=bool(cond3),
        passed=bool(cond1 and cond2 and cond3),
    )
    json.dump(res, open(os.path.join(ctx.rd, "gate2.json"), "w"), indent=2)
    np.savez(os.path.join(ctx.rd, "clean.npz"), **{k: v for k, v in c.items() if v is not None})
    print(
        f"[{ctx.ds}/{ctx.arch}] GATE 2: acc {acc}  agreement QSim/INT8 {agree:.4f}  int ops {ops}  -> "
        f"{'PASS' if res['passed'] else '*** FAIL ***'}"
    )
    # Hard stop (even when strict_gates=False): attacks need images that every precision classifies correctly.
    allok = np.all([c[k] == y for k in ("fp32", "fp16", "int8") if c.get(k) is not None], axis=0)
    if allok.sum() < 20:
        hint = (
            (
                "FP32 accuracy is near chance -> the FINE-TUNING went wrong (delete fp32_final.pt / ft_checkpoint.pt in this "
                "pair's folder, then check the epoch log)."
            )
            if acc["fp32"] < 0.30
            else (
                "INT8 accuracy collapsed -> naive 8-bit quantisation is breaking this model. Run "
                "vg.diagnose_quantisation(ctx) to see which layers cause it."
            )
        )
        raise GateError(
            f"Only {int(allok.sum())} of {len(y)} evaluation images are classified correctly by all precisions "
            f"(accuracies: {acc}). {hint}"
        )
    if not res["passed"] and cfg.strict_gates:
        why = [
            n
            for n, ok in (
                ("INT8 accuracy drop > tolerance", cond1),
                ("QSim/INT8 agreement too low", cond2),
                ("no integer kernels in profiler", cond3),
            )
            if not ok
        ]
        raise GateError(
            "Gate 2 failed: " + "; ".join(why) + ". Revise calibration (calib_n, observers, excluded ops) "
            "and document the change before attacking."
        )
    return res


def diagnose_quantisation(ctx: Ctx, top_k: int = 12, n: int = 200) -> pd.DataFrame:
    """Find out WHY INT8 loses accuracy. Reports accuracy with (a) all fake-quant on, (b) weights only,
    (c) activations only, then ranks activation quantisers by how much accuracy returns when that single
    one is left in float. Uses the QSim twin (same scales as INT8), on the GPU."""
    dev = ctx.device
    x, y = ctx.x[:n], ctx.y[:n].numpy()
    bb = ctx.qsim.backbone
    fqs = [(nm, m) for nm, m in bb.named_modules() if isinstance(m, FakeQuantize)]
    is_w = lambda m: bool(getattr(m, "is_per_channel", False))
    # human-readable names: which graph node does each activation quantiser follow?
    label = {}
    for node in bb.graph.nodes:
        if node.op == "call_module" and node.target in dict(fqs) and node.args:
            label[node.target] = str(node.args[0])
    acc = lambda: float((predict(ctx.qsim, x, 100, dev) == y).mean())

    def setting(weights: bool, acts: bool, off: Optional[str] = None):
        for nm, m in fqs:
            on = (weights if is_w(m) else acts) and nm != off
            (m.enable_fake_quant if on else m.disable_fake_quant)()

    rows = [dict(setting="FP32 (no quantisation)", acc=float((predict(ctx.fp32, x, 100, dev) == y).mean()))]
    for name, w, a in (
        ("QSim: weights + activations (= INT8 scales)", True, True),
        ("QSim: weights only", True, False),
        ("QSim: activations only", False, True),
    ):
        setting(w, a)
        rows.append(dict(setting=name, acc=acc()))
    base = rows[1]["acc"]
    gains = []
    for nm, m in fqs:
        if is_w(m):
            continue
        setting(True, True, off=nm)
        gains.append((acc() - base, nm))
    setting(True, True)  # restore
    for g, nm in sorted(gains, reverse=True)[:top_k]:
        rows.append(dict(setting=f"leave in float: {label.get(nm, nm)}", acc=base + g, gain_vs_full_quant=g))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# Attacks
# ----------------------------------------------------------------------------


def _attack(kind: str, model: nn.Module, eps255: int, cfg: Config, steps: Optional[int] = None):
    import torchattacks

    eps = eps255 / 255.0
    if kind == "FGSM":
        return torchattacks.FGSM(model, eps=eps)
    steps = steps or cfg.pgd_steps
    return torchattacks.PGD(model, eps=eps, alpha=cfg.pgd_alpha_mult * eps / steps, steps=steps, random_start=True)


def generate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    kind: str,
    eps255: int,
    cfg: Config,
    device,
    seed: int,
    steps: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    atk = _attack(kind, model, eps255, cfg, steps)
    out = []
    for b, i in enumerate(range(0, len(x), cfg.attack_bs)):
        torch.manual_seed(seed * 100003 + eps255 * 1009 + b)
        torch.cuda.manual_seed_all(seed * 100003 + eps255 * 1009 + b)
        out.append(atk(x[i : i + cfg.attack_bs].to(device), y[i : i + cfg.attack_bs].to(device)).detach().cpu())
    return torch.cat(out)


def verify_adv(x: torch.Tensor, xa: torch.Tensor, eps255: int) -> Tuple[float, float]:
    d = xa - x
    linf = d.flatten(1).abs().max(1).values
    assert float(linf.max()) <= eps255 / 255.0 + 1e-6, f"l_inf violated: {float(linf.max())} > {eps255 / 255}"
    assert float(xa.min()) >= 0.0 and float(xa.max()) <= 1.0, "adversarial image outside [0,1]"
    return float(linf.mean()), float(d.flatten(1).norm(dim=1).mean())


def run_attacks(
    ctx: Ctx,
    kinds=("FGSM", "PGD"),
    eps_list: Optional[List[int]] = None,
    seed: Optional[int] = None,
    tag: str = "",
    out_sub: str = "attacks",
) -> None:
    """Scenarios
    A0: crafted on FP32, evaluated on FP32                         (white-box reference)
    A1: crafted on QSim,  evaluated on INT8   (and crafted on FP16, evaluated on FP16)
    A2: crafted on FP32, evaluated on FP16 / INT8                  (grey-box transfer)
    """
    cfg, dev = ctx.cfg, ctx.device
    seed = cfg.seed if seed is None else seed
    eps_list = list(eps_list) if eps_list is not None else list(cfg.eps_255) + [cfg.diag_eps_255]
    d = os.path.join(ctx.rd, out_sub)
    os.makedirs(d, exist_ok=True)
    cpu = torch.device("cpu")
    for kind in kinds:
        for e in eps_list:
            f = os.path.join(d, f"{kind}_eps{e:03d}{tag}.npz")
            if os.path.exists(f):
                continue
            t0 = time.time()
            r = {}
            xa32 = generate(ctx.fp32, ctx.x, ctx.y, kind, e, cfg, dev, seed)
            r["linf32"], r["l2_32"] = verify_adv(ctx.x, xa32, e)
            r["A0_fp32"] = predict(ctx.fp32, xa32, 250, dev)  # A0
            if ctx.fp16 is not None:
                r["A2_fp16"] = predict(ctx.fp16, xa32, 250, dev)  # A2 -> FP16
                xa16 = generate(ctx.fp16, ctx.x, ctx.y, kind, e, cfg, dev, seed)
                r["linf16"], r["l2_16"] = verify_adv(ctx.x, xa16, e)
                r["A1_fp16"] = predict(ctx.fp16, xa16, 250, dev)  # A1 (FP16 arm)
            if ctx.int8 is not None:
                r["A2_int8"] = predict(ctx.int8, xa32, cfg.int8_bs, cpu)  # A2 -> INT8
                xaq = generate(ctx.qsim, ctx.x, ctx.y, kind, e, cfg, dev, seed)
                r["linfq"], r["l2_q"] = verify_adv(ctx.x, xaq, e)
                r["A1_qsim"] = predict(ctx.qsim, xaq, 250, dev)  # proxy success (fidelity check)
                r["A1_int8"] = predict(ctx.int8, xaq, cfg.int8_bs, cpu)  # A1 measured on real INT8
            np.savez(f, y=ctx.y.numpy(), eps255=e, **{k: np.asarray(v) for k, v in r.items()})
            print(f"[{ctx.ds}/{ctx.arch}] {kind} eps={e}/255 done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------
# Diagnostics D4, D5, K=50
# ----------------------------------------------------------------------------


def grad_zero_fraction(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device, tol: float, bs: int = 50) -> dict:
    model.eval()
    mx = []
    for i in range(0, len(x), bs):
        xb = x[i : i + bs].to(device).clone().requires_grad_(True)
        F.cross_entropy(model(xb), y[i : i + bs].to(device), reduction="sum").backward()
        mx.append(xb.grad.detach().abs().flatten(1).max(1).values.cpu())
    m = torch.cat(mx)
    return dict(
        frac_exact_zero=float((m == 0).float().mean()),
        frac_below_tol=float((m <= tol).float().mean()),
        median_max_grad=float(m.median()),
    )


def pgd_custom(
    model, x, y, eps255, cfg: Config, loss: str = "ce", T: float = 1.0, steps: Optional[int] = None, seed: int = 0
):
    """Sign-gradient PGD with random start and selectable loss: ce | margin | ce_temp (logits / T)."""
    steps = steps or cfg.pgd_steps
    eps = eps255 / 255.0
    alpha = cfg.pgd_alpha_mult * eps / steps
    g = torch.Generator(device=x.device).manual_seed(seed)
    xa = (x + (torch.rand(x.shape, generator=g, device=x.device) * 2 - 1) * eps).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        z = model(xa)
        if loss == "margin":
            zy = z.gather(1, y[:, None])[:, 0]
            zo = z.masked_fill(F.one_hot(y, z.shape[1]).bool(), -1e9).max(1).values
            l = (zo - zy).sum()
        else:
            l = F.cross_entropy(z / (T if loss == "ce_temp" else 1.0), y, reduction="sum")
        (gr,) = torch.autograd.grad(l, xa)
        xa = xa.detach() + alpha * gr.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa.detach()


def run_diagnostics(ctx: Ctx) -> dict:
    """D4 (zero-gradient fractions) and D5 (alternative losses, worst-case ASR) on the first d5_n eval images."""
    cfg, dev = ctx.cfg, ctx.device
    f = os.path.join(ctx.rd, "diagnostics.json")
    if os.path.exists(f):
        return json.load(open(f))
    c = ctx.clean or clean_predictions(ctx)
    n = min(cfg.d5_n, len(ctx.x))
    x, y = ctx.x[:n], ctx.y[:n]
    S = np.where(
        np.all(
            [c[k][:n] == c["y"][:n] for k in ("fp32", "int8") + (("fp16",) if c.get("fp16") is not None else ())],
            axis=0,
        )
    )[0]
    res = dict(n=int(n), n_matched=int(len(S)), D4={}, D5={})
    res["D4"]["fp32"] = grad_zero_fraction(ctx.fp32, x[S], y[S], dev, cfg.grad_tol)
    res["D4"]["qsim"] = grad_zero_fraction(ctx.qsim, x[S], y[S], dev, cfg.grad_tol)
    if ctx.fp16 is not None:
        res["D4"]["fp16"] = grad_zero_fraction(ctx.fp16, x[S], y[S], dev, cfg.grad_tol)
    variants = [("ce", 1.0), ("margin", 1.0), ("ce_temp", cfg.temp_T)]
    cpu = torch.device("cpu")
    for e in cfg.d5_eps_255:
        per = {}
        for tgt, mdl, evalm, evdev in (("A0_fp32", ctx.fp32, ctx.fp32, dev), ("A1_int8", ctx.qsim, ctx.int8, cpu)):
            succ = {}
            for name, T in variants:
                adv = torch.cat(
                    [
                        pgd_custom(
                            mdl,
                            x[S][i : i + cfg.attack_bs].to(dev),
                            y[S][i : i + cfg.attack_bs].to(dev),
                            e,
                            cfg,
                            name,
                            T,
                            seed=cfg.seed + i,
                        ).cpu()
                        for i in range(0, len(S), cfg.attack_bs)
                    ]
                )
                verify_adv(x[S], adv, e)
                p = predict(evalm, adv, cfg.int8_bs if evdev.type == "cpu" else 250, evdev)
                succ[name] = p != y[S].numpy()
            worst = np.any(np.stack(list(succ.values())), axis=0)
            per[tgt] = {**{k: float(v.mean()) for k, v in succ.items()}, "worst_case": float(worst.mean())}
        res["D5"][f"eps{e}"] = per
    json.dump(res, open(f, "w"), indent=2)
    print(f"[{ctx.ds}/{ctx.arch}] diagnostics saved ->", f)
    return res


def run_k50_check(ctx: Ctx) -> dict:
    """PGD K=50 vs K=20 on a subset: is 20 steps enough?"""
    cfg, dev = ctx.cfg, ctx.device
    f = os.path.join(ctx.rd, "k50_check.json")
    if os.path.exists(f):
        return json.load(open(f))
    n = min(cfg.k50_n, len(ctx.x))
    x, y = ctx.x[:n], ctx.y[:n]
    res = {}
    for e in cfg.k50_eps_255:
        row = {}
        for K in (cfg.pgd_steps, 50):
            xa = generate(ctx.fp32, x, y, "PGD", e, cfg, dev, cfg.seed, steps=K)
            row[f"A0_K{K}"] = float((predict(ctx.fp32, xa, 250, dev) != y.numpy()).mean())
            if ctx.qsim is not None:
                xq = generate(ctx.qsim, x, y, "PGD", e, cfg, dev, cfg.seed, steps=K)
                row[f"A1_K{K}"] = float((predict(ctx.int8, xq, cfg.int8_bs, torch.device("cpu")) != y.numpy()).mean())
        res[f"eps{e}"] = row
    json.dump(res, open(f, "w"), indent=2)
    print(f"[{ctx.ds}/{ctx.arch}] K=50 check:", res)
    return res


# ----------------------------------------------------------------------------
# Efficiency
# ----------------------------------------------------------------------------


def _time(fn, warmup, iters, sync) -> Tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    q = np.percentile(ts, [25, 50, 75])
    return float(q[1]), float(q[0]), float(q[2])


def measure_efficiency(ctx: Ctx) -> pd.DataFrame:
    cfg = ctx.cfg
    f = os.path.join(ctx.rd, "efficiency.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    rows = []
    H = ctx.x.shape[-1]
    sizes = {"fp32": model_size_mb(ctx.fp32), "int8": model_size_mb(ctx.int8)}
    if ctx.fp16 is not None:
        sizes["fp16"] = model_size_mb(ctx.fp16)
    cpu = torch.device("cpu")
    fp32_cpu = copy.deepcopy(ctx.fp32).cpu().eval()
    jobs = [("fp32", "cpu", fp32_cpu), ("int8", "cpu", ctx.int8)]
    if ctx.device.type == "cuda":
        jobs += [("fp32", "gpu", ctx.fp32), ("fp16", "gpu", ctx.fp16)]
    for prec, dev, m in jobs:
        d = cpu if dev == "cpu" else ctx.device
        for bs in cfg.lat_batches:
            xb = torch.rand(bs, 3, H, H, device=d)
            with torch.no_grad():
                med, lo, hi = _time(lambda: m(xb), cfg.warmup, cfg.iters, sync=(dev == "gpu"))
            rows.append(
                dict(
                    dataset=ctx.ds,
                    arch=ctx.arch,
                    precision=prec,
                    device=dev,
                    batch=bs,
                    median_ms=med,
                    q25_ms=lo,
                    q75_ms=hi,
                    size_mb=sizes[prec],
                    threads=torch.get_num_threads() if dev == "cpu" else None,
                    compression_vs_fp32=sizes["fp32"] / sizes[prec],
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# Pipeline orchestration
# ----------------------------------------------------------------------------


def run_pipeline(cfg: Config, ds: str, arch: str, data=None, stages=("attacks", "diagnostics", "k50", "efficiency")):
    ctx = setup_context(cfg, ds, arch, data=data)
    json.dump(env_info(), open(os.path.join(ctx.rd, "env.json"), "w"), indent=2)
    clean_predictions(ctx)
    gate2(ctx)
    if "attacks" in stages:
        run_attacks(ctx)
    if "diagnostics" in stages:
        run_diagnostics(ctx)
    if "k50" in stages:
        run_k50_check(ctx)
    if "efficiency" in stages:
        measure_efficiency(ctx)
    return ctx


def free(ctx: Optional[Ctx] = None):
    import gc

    del ctx
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Multi-seed study (calibration draw x PGD random start)
# ----------------------------------------------------------------------------


def run_seed_study(cfg: Config, ds: str, arch: str, data=None) -> pd.DataFrame:
    f = os.path.join(cfg.run_dir(ds, arch), "seed_study.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    rows = []
    for s in cfg.seeds:
        ctx = setup_context(cfg, ds, arch, data=data, calib_seed=s, state_suffix=f"_seed{s}")
        c = clean_predictions(ctx)
        ok = (c["fp32"] == c["y"]) & (c["int8"] == c["y"])
        run_attacks(ctx, kinds=("PGD",), eps_list=list(cfg.seed_eps_255), seed=s, tag=f"_s{s}", out_sub="seed_study")
        for e in cfg.seed_eps_255:
            r = np.load(os.path.join(ctx.rd, "seed_study", f"PGD_eps{e:03d}_s{s}.npz"))
            rows.append(
                dict(
                    seed=s,
                    eps255=e,
                    clean_acc_int8=float((c["int8"] == c["y"]).mean()),
                    asr_A1_int8=float((r["A1_int8"][ok] != r["y"][ok]).mean()),
                    asr_A2_int8=float((r["A2_int8"][ok] != r["y"][ok]).mean()),
                    asr_A0_fp32=float((r["A0_fp32"][ok] != r["y"][ok]).mean()),
                )
            )
        free(ctx)
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    g = df.groupby("eps255").agg(["mean", "min", "max"]).round(4)
    print(g.to_string())
    return df


# ----------------------------------------------------------------------------
# Additional checks: FP16 gradient underflow, stronger attack (APGD-CE)
# ----------------------------------------------------------------------------


def _matched_idx(ctx: Ctx, n: int):
    """Indices (within the first n eval images) that every available precision classifies correctly."""
    c = clean_predictions(ctx)
    n = min(n, len(ctx.x))
    ok = [c[k][:n] == c["y"][:n] for k in ("fp32", "int8", "fp16") if c.get(k) is not None]
    return np.where(np.all(ok, axis=0))[0], n


def _pgd_with_loss(model, x, y, eps255, cfg, dev, reduction="mean", scale=1.0, seed=0):
    """PGD (random start, cfg.pgd_steps) with a chosen loss reduction and loss scale (for the FP16 check)."""
    eps = eps255 / 255.0
    alpha = cfg.pgd_alpha_mult * eps / cfg.pgd_steps
    g = torch.Generator(device=dev).manual_seed(seed)
    xa = (x + (torch.rand(x.shape, generator=g, device=dev) * 2 - 1) * eps).clamp(0, 1)
    nonfinite = 0.0
    for _ in range(cfg.pgd_steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y, reduction=reduction) * scale
        (gr,) = torch.autograd.grad(loss, xa)
        nonfinite += float((~torch.isfinite(gr)).float().mean())
        gr = torch.nan_to_num(gr, nan=0.0, posinf=0.0, neginf=0.0)
        xa = xa.detach() + alpha * gr.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa.detach(), nonfinite / cfg.pgd_steps


def fp16_check(ctx: Ctx, n: int = 200, eps255: int = 2) -> pd.DataFrame:
    """Why do naive white-box attacks on FP16 fail?

    Reports (a) the share of input pixels with an exactly-zero gradient and (b) PGD success at eps255,
    for FP32 and FP16 under three losses: batch-mean CE, summed CE, summed CE x 1024.
    NOTE: this check bypasses HalfNet's built-in scaling by calling the wrapped net through a plain
    forward, so the 'mean' rows show the unscaled behaviour.
    """
    cfg, dev = ctx.cfg, ctx.device
    S, n = _matched_idx(ctx, n)
    x, y = ctx.x[:n][S].to(dev), ctx.y[:n][S].to(dev)
    bs = cfg.attack_bs
    models = {"fp32": ctx.fp32}
    if ctx.fp16 is not None:
        half = ctx.fp16.net  # the raw FP16 network, without the loss-scaling hooks
        models["fp16"] = lambda z: half(z.half()).float()
    rows = []
    for name, m in models.items():
        for red, sc in (("mean", 1.0), ("sum", 1.0), ("sum", 1024.0)):
            xb = x[:bs].clone().requires_grad_(True)
            (gr,) = torch.autograd.grad(F.cross_entropy(m(xb), y[:bs], reduction=red) * sc, xb)
            rows.append(
                dict(
                    kind="zero_pixel_fraction",
                    model=name,
                    loss=f"{red} x{sc:g}",
                    value=float((gr == 0).float().mean()),
                    nonfinite=float((~torch.isfinite(gr)).float().mean()),
                )
            )
    for name, m in models.items():
        for red, sc in (("mean", 1.0), ("sum", 1.0), ("sum", 1024.0)):
            adv, bad = [], 0.0
            for i in range(0, len(x), bs):
                a, b = _pgd_with_loss(m, x[i : i + bs], y[i : i + bs], eps255, cfg, dev, red, sc, seed=cfg.seed + i)
                adv.append(a)
                bad += b
            adv = torch.cat(adv)
            verify_adv(x.cpu(), adv.cpu(), eps255)
            with torch.no_grad():
                pred = torch.cat([m(adv[i : i + bs]).argmax(1) for i in range(0, len(adv), bs)])
            rows.append(
                dict(
                    kind=f"PGD_ASR@{eps255}/255",
                    model=name,
                    loss=f"{red} x{sc:g}",
                    value=float((pred != y).float().mean()),
                    nonfinite=bad,
                )
            )
    return pd.DataFrame(rows).assign(dataset=ctx.ds, arch=ctx.arch, n_matched=len(S))


def apgd_check(ctx: Ctx, eps_list=(2, 4), n: int = 500, steps: int = 50) -> pd.DataFrame:
    """Stronger-attack check: APGD-CE (main component of AutoAttack) vs. our PGD-20, FP32 vs. INT8."""
    import torchattacks

    cfg, dev, cpu = ctx.cfg, ctx.device, torch.device("cpu")
    S, n = _matched_idx(ctx, n)
    x, y = ctx.x[:n][S], ctx.y[:n][S]
    rows = []
    for e in eps_list:
        for tgt, mdl, evalm, evdev in (("A0_fp32", ctx.fp32, ctx.fp32, dev), ("A1_int8", ctx.qsim, ctx.int8, cpu)):
            mdl.eval()
            atk = torchattacks.APGD(
                mdl, norm="Linf", eps=e / 255.0, steps=steps, n_restarts=1, loss="ce", seed=cfg.seed
            )
            adv = torch.cat(
                [
                    atk(x[i : i + cfg.attack_bs].to(dev), y[i : i + cfg.attack_bs].to(dev)).detach().cpu()
                    for i in range(0, len(S), cfg.attack_bs)
                ]
            )
            verify_adv(x, adv, e)
            bs = cfg.int8_bs if evdev.type == "cpu" else 250
            asr_apgd = float((predict(evalm, adv, bs, evdev) != y.numpy()).mean())
            pgd = torch.cat(
                [
                    pgd_custom(
                        mdl,
                        x[i : i + cfg.attack_bs].to(dev),
                        y[i : i + cfg.attack_bs].to(dev),
                        e,
                        cfg,
                        "ce",
                        seed=cfg.seed + i,
                    ).cpu()
                    for i in range(0, len(S), cfg.attack_bs)
                ]
            )
            asr_pgd = float((predict(evalm, pgd, bs, evdev) != y.numpy()).mean())
            rows.append(
                dict(
                    dataset=ctx.ds,
                    arch=ctx.arch,
                    eps255=e,
                    target=tgt,
                    n_matched=len(S),
                    pgd20_ce=asr_pgd,
                    apgd50_ce=asr_apgd,
                )
            )
    return pd.DataFrame(rows)


def run_over_pairs(cfg: Config, fn, csv_name: str) -> pd.DataFrame:
    """Run fn(ctx) for every (dataset, architecture) pair, save the concatenated result to analysis/<csv_name>."""
    out, cache = [], {}
    for ds, arch in cfg.all_pairs():
        if ds not in cache:
            cache.clear()
            cache[ds] = load_dataset_arrays(cfg, ds)
        ctx = setup_context(cfg, ds, arch, data=cache[ds])
        out.append(fn(ctx))
        print(out[-1].to_string(index=False))
        free(ctx)
    res = pd.concat(out)
    d = os.path.join(cfg.out_dir, "analysis")
    os.makedirs(d, exist_ok=True)
    res.to_csv(os.path.join(d, csv_name), index=False)
    print("saved ->", os.path.join(d, csv_name))
    return res


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

# scenario label -> (npz key, target precision)
SCEN = {
    ("A0", "fp32"): "A0_fp32",
    ("A1", "int8"): "A1_int8",
    ("A1", "fp16"): "A1_fp16",
    ("A2", "int8"): "A2_int8",
    ("A2", "fp16"): "A2_fp16",
}


def tipping_point(eps255: np.ndarray, asr: np.ndarray, tau: float) -> float:
    e = np.concatenate([[0.0], eps255.astype(float)])
    a = np.concatenate([[0.0], asr])
    hit = np.where(a >= tau)[0]
    if len(hit) == 0:
        return float("nan")
    i = hit[0]
    if i == 0:
        return 0.0
    return float(e[i - 1] + (tau - a[i - 1]) * (e[i] - e[i - 1]) / max(a[i] - a[i - 1], 1e-12))


def load_pair(cfg: Config, ds: str, arch: str) -> Optional[dict]:
    rd = cfg.run_dir(ds, arch)
    if not os.path.exists(os.path.join(rd, "clean.npz")):
        return None
    clean = dict(np.load(os.path.join(rd, "clean.npz"), allow_pickle=True))
    y = clean["y"]
    precs = [k for k in ("fp32", "fp16", "int8") if k in clean]
    S = np.all([clean[k] == y for k in precs], axis=0)
    data = {}
    for f in sorted(glob.glob(os.path.join(rd, "attacks", "*.npz"))):
        z = np.load(f)
        kind = os.path.basename(f).split("_")[0]
        data[(kind, int(z["eps255"]))] = {k: z[k] for k in z.files}
    return dict(clean=clean, y=y, S=S, data=data)


def analyse(cfg: Config, make_plots: bool = True) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    asr_rows, mc_rows, tip_rows, diag_rows = [], [], [], []
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    curves = {}
    for ds in cfg.datasets:
        for arch in cfg.archs:
            if not os.path.isdir(os.path.join(cfg.out_dir, f"{ds}_{arch}")):
                continue  # pair not run (yet) / lives in another person's folder
            P = load_pair(cfg, ds, arch)
            if P is None or not P["data"]:
                continue
            y, S = P["y"], P["S"]
            n = int(S.sum())
            for kind in ("FGSM", "PGD"):
                eps_all = sorted(e for (k, e) in P["data"] if k == kind)
                eps_main = [e for e in eps_all if e != cfg.diag_eps_255]
                succ = {}  # (scenario, target) -> bool matrix [len(eps_all), n]
                for (sc, tg), key in SCEN.items():
                    if all(key in P["data"][(kind, e)] for e in eps_all):
                        succ[(sc, tg)] = np.stack([(P["data"][(kind, e)][key] != y)[S] for e in eps_all])
                if not succ:
                    continue
                B = rng.integers(0, n, size=(cfg.n_boot, n))
                for (sc, tg), M in succ.items():
                    boot = M[:, B].mean(-1)  # [E, n_boot]
                    for j, e in enumerate(eps_all):
                        lo, hi = np.percentile(boot[j], [2.5, 97.5])
                        full = P["data"][(kind, e)][SCEN[(sc, tg)]] != y
                        asr_rows.append(
                            dict(
                                dataset=ds,
                                arch=arch,
                                attack=kind,
                                eps255=e,
                                scenario=sc,
                                target=tg,
                                n_matched=n,
                                asr=M[j].mean(),
                                ci_lo=lo,
                                ci_hi=hi,
                                robust_acc=1 - full.mean(),
                                diagnostic_eps=(e == cfg.diag_eps_255),
                            )
                        )
                    curves[(ds, arch, kind, sc, tg)] = (
                        np.array(eps_all),
                        M.mean(1),
                        np.percentile(boot, 2.5, axis=1),
                        np.percentile(boot, 97.5, axis=1),
                    )
                # --- paired McNemar vs FP32 white-box baseline (A0), Holm across eps
                if ("A0", "fp32") in succ:
                    base = succ[("A0", "fp32")]
                    for (sc, tg), M in succ.items():
                        if (sc, tg) == ("A0", "fp32"):
                            continue
                        ps, recs = [], []
                        for j, e in enumerate(eps_all):
                            a, b = base[j], M[j]
                            tab = [
                                [int((a & b).sum()), int((a & ~b).sum())],
                                [int((~a & b).sum()), int((~a & ~b).sum())],
                            ]
                            p = mcnemar(tab, exact=True).pvalue
                            ps.append(p)
                            recs.append((e, tab))
                        ph = multipletests(ps, method="holm")[1]
                        for (e, tab), p, pp in zip(recs, ps, ph):
                            mc_rows.append(
                                dict(
                                    dataset=ds,
                                    arch=arch,
                                    attack=kind,
                                    eps255=e,
                                    compare=f"{sc}/{tg} vs A0/fp32",
                                    both=tab[0][0],
                                    only_fp32=tab[0][1],
                                    only_compressed=tab[1][0],
                                    neither=tab[1][1],
                                    p_raw=p,
                                    p_holm=pp,
                                    delta_asr=float(M[eps_all.index(e)].mean() - base[eps_all.index(e)].mean()),
                                )
                            )
                # --- tipping points (main grid only), bootstrap CI and shift vs FP32
                mi = [eps_all.index(e) for e in eps_main]
                em = np.array(eps_main)
                for tau in cfg.taus:
                    tp = {}
                    for key, M in succ.items():
                        mm = M[mi]
                        pt = tipping_point(em, mm.mean(1), tau)
                        bt = np.array([tipping_point(em, mm[:, b].mean(1), tau) for b in B[:200]])
                        tp[key] = (pt, bt)
                        lo, hi = np.nanpercentile(bt, [2.5, 97.5]) if np.isfinite(bt).any() else (np.nan, np.nan)
                        tip_rows.append(
                            dict(
                                dataset=ds,
                                arch=arch,
                                attack=kind,
                                scenario=key[0],
                                target=key[1],
                                tau=tau,
                                eps_star_255=pt,
                                ci_lo=lo,
                                ci_hi=hi,
                                reached=bool(np.isfinite(pt)),
                            )
                        )
                    if ("A0", "fp32") in tp:
                        for key in tp:
                            if key == ("A0", "fp32"):
                                continue
                            d = tp[("A0", "fp32")][0] - tp[key][0]
                            tip_rows.append(
                                dict(
                                    dataset=ds,
                                    arch=arch,
                                    attack=kind,
                                    scenario=key[0],
                                    target=key[1],
                                    tau=tau,
                                    eps_star_255=d,
                                    ci_lo=np.nan,
                                    ci_hi=np.nan,
                                    reached=bool(np.isfinite(d)),
                                    note="shift: eps*(FP32 A0) - eps*(this)",
                                )
                            )
                # --- D1 / D2 / D3
                for key, M in succ.items():
                    a = M.mean(1)
                    diag_rows.append(
                        dict(
                            dataset=ds,
                            arch=arch,
                            attack=kind,
                            check="D1_monotone",
                            scenario=key[0],
                            target=key[1],
                            violations=int((np.diff(a) < -1e-12).sum()),
                            detail=f"ASR@{eps_all[-1]}/255={a[-1]:.3f}",
                        )
                    )
            # D2: PGD >= FGSM (per scenario/target/eps) ; D3: white-box >= transfer on same target
            df = pd.DataFrame([r for r in asr_rows if r["dataset"] == ds and r["arch"] == arch])
            if not df.empty:
                piv = df.pivot_table(index=["scenario", "target", "eps255"], columns="attack", values="asr")
                if {"FGSM", "PGD"} <= set(piv.columns):
                    v = piv[piv["PGD"] < piv["FGSM"] - 1e-12]
                    diag_rows.append(
                        dict(
                            dataset=ds,
                            arch=arch,
                            attack="both",
                            check="D2_PGD>=FGSM",
                            scenario="all",
                            target="all",
                            violations=len(v),
                            detail="; ".join(
                                f"{i[0]}/{i[1]}@{i[2]} (PGD {r.PGD:.3f} < FGSM {r.FGSM:.3f})" for i, r in v.iterrows()
                            )[:500],
                        )
                    )
                for tg in ("int8", "fp16"):
                    for kind in ("FGSM", "PGD"):
                        a1 = df[(df.attack == kind) & (df.scenario == "A1") & (df.target == tg)].set_index("eps255").asr
                        a2 = df[(df.attack == kind) & (df.scenario == "A2") & (df.target == tg)].set_index("eps255").asr
                        if len(a1) and len(a2):
                            v = a1 < a2 - 1e-12
                            diag_rows.append(
                                dict(
                                    dataset=ds,
                                    arch=arch,
                                    attack=kind,
                                    check="D3_whitebox>=transfer",
                                    scenario="A1 vs A2",
                                    target=tg,
                                    violations=int(v.sum()),
                                    detail=", ".join(f"{e}/255" for e in v[v].index),
                                )
                            )
    out = dict(
        asr=pd.DataFrame(asr_rows),
        mcnemar=pd.DataFrame(mc_rows),
        tipping=pd.DataFrame(tip_rows),
        checks=pd.DataFrame(diag_rows),
    )
    ad = os.path.join(cfg.out_dir, "analysis")
    os.makedirs(ad, exist_ok=True)
    for k, v in out.items():
        v.to_csv(os.path.join(ad, f"{k}.csv"), index=False)
    # D4 / D5 / efficiency / gates roll-ups
    roll = {"gate1": [], "gate2": [], "d4": [], "d5": [], "k50": [], "eff": []}
    for ds in cfg.datasets:
        for arch in cfg.archs:
            if not os.path.isdir(os.path.join(cfg.out_dir, f"{ds}_{arch}")):
                continue  # pair not run (yet) / lives in another person's folder
            rd = cfg.run_dir(ds, arch)
            for nm, fn in (("gate1", "gate1.json"), ("gate2", "gate2.json")):
                if os.path.exists(os.path.join(rd, fn)):
                    j = json.load(open(os.path.join(rd, fn)))
                    roll[nm].append(pd.json_normalize(j))
            if os.path.exists(os.path.join(rd, "diagnostics.json")):
                j = json.load(open(os.path.join(rd, "diagnostics.json")))
                for m, v in j["D4"].items():
                    roll["d4"].append(dict(dataset=ds, arch=arch, model=m, **v))
                for e, per in j["D5"].items():
                    for t, v in per.items():
                        roll["d5"].append(dict(dataset=ds, arch=arch, eps=e, scenario=t, **v))
            if os.path.exists(os.path.join(rd, "k50_check.json")):
                for e, v in json.load(open(os.path.join(rd, "k50_check.json"))).items():
                    roll["k50"].append(dict(dataset=ds, arch=arch, eps=e, **v))
            if os.path.exists(os.path.join(rd, "efficiency.csv")):
                roll["eff"].append(pd.read_csv(os.path.join(rd, "efficiency.csv")))
    for k, v in roll.items():
        if v:
            df = pd.concat(v) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v)
            out[k] = df
            df.to_csv(os.path.join(ad, f"{k}.csv"), index=False)
    if make_plots:
        plot_curves(cfg, curves, ad)
    return out


def plot_curves(cfg: Config, curves: dict, outdir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        ("A0", "fp32"): ("FP32 white-box (A0)", "k", "-"),
        ("A1", "int8"): ("INT8 white-box via QSim (A1)", "C3", "-"),
        ("A2", "int8"): ("INT8 transfer from FP32 (A2)", "C3", "--"),
        ("A1", "fp16"): ("FP16 white-box (A1)", "C0", "-"),
        ("A2", "fp16"): ("FP16 transfer from FP32 (A2)", "C0", "--"),
    }
    for ds in cfg.datasets:
        for arch in cfg.archs:
            if not os.path.isdir(os.path.join(cfg.out_dir, f"{ds}_{arch}")):
                continue  # pair not run (yet) / lives in another person's folder
            fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            any_ = False
            for ax, kind in zip(axs, ("FGSM", "PGD")):
                for (d, a, k, sc, tg), (e, m, lo, hi) in curves.items():
                    if (d, a, k) != (ds, arch, kind):
                        continue
                    lab, col, ls = style[(sc, tg)]
                    ax.plot(e, m * 100, ls, color=col, marker="o", ms=3, label=lab)
                    ax.fill_between(e, lo * 100, hi * 100, color=col, alpha=0.12)
                    any_ = True
                ax.set_xscale("symlog", linthresh=4)
                ax.set_xlabel("epsilon (x/255)")
                ax.set_title(f"{kind} - {ds} / {arch}")
                ax.grid(alpha=0.3)
            axs[0].set_ylabel("ASR on matched set (%)")
            if any_:
                axs[1].legend(fontsize=7, loc="lower right")
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, f"asr_{ds}_{arch}.png"), dpi=150)
            plt.close(fig)
