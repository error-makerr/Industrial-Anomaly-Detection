# EdgeNetV4 — Multi-Task Industrial Defect Inspection

<p align="center">
  <img src="results/figures/end_to_end_pipe.png" width="90%" alt="EdgeNetV4 End-to-End Pipeline"/>
</p>

> **Three inspection questions. One forward pass. One model.**
>
> Is it defective? → What type of defect? → Which product?

EdgeNetV4 is a multi-task deep learning framework that simultaneously performs **binary defect detection**, **8-class defect type classification**, and **17-class product identification** on industrial inspection images — all in a single forward pass under 20ms on consumer hardware.

**This is the first framework to perform all three tasks simultaneously on a merged multi-source industrial dataset.** No prior published work solves this combination.

---

## Highlights

| Metric | Validation | Test |
|---|---|---|
| Defect Macro F1 | **0.9662** | **0.9570** |
| Binary Accuracy | 99.3% | — |
| Binary AUC-ROC | 0.9978 | 0.9964 |
| Product Accuracy | 100.0% | — |
| Parameters | 7.88M | — |
| Inference | <20ms (RTX 3060) | — |

Every defect class achieves F1 > 0.91, including `minor_defect` (F1 = 0.940) with only 196 training samples.

---

## Architecture

<p align="center">
  <img src="results/figures/architecture.png" width="85%" alt="EdgeNetV4 Architecture"/>
</p>

EdgeNetV4 uses a **compound-scaled MBConv backbone with squeeze-and-excitation attention (B2 configuration)** exposing three feature scales:

| Feature Map | Channels | Stride | Captures |
|---|---|---|---|
| f2 | 48 | 8 | Fine texture (scratches, surface grain) |
| f3 | 120 | 16 | Mid-level shape (crack geometry) |
| f4 | 352 | 32 | Semantic identity (defect category) |

**Key innovations:**

1. **Multi-scale defect head** — concatenates GAP(f2) + GAP(f3) + GAP(f4) into a 520-dimensional descriptor. Ablation shows this contributes +0.062 F1 over single-scale design.
2. **Identity-initialised CoordAttention** — weight=0, bias=+10 forces gates to ≈1.0 at init, preserving pretrained features through early training.
3. **Per-class focal loss with inverse-frequency weighting** — per-class γ ∈ [2.0, 4.0] targets the hardest classes; lifted `minor_defect` from F1 < 0.40 to 0.940.

---

## Dataset

A unified multi-domain dataset of **21,344 images** merged from three public industrial benchmarks:

| Source | Images | Product Categories | Defect Scale |
|---|---|---|---|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 5,354 | 15 | 10–50 px |
| [Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | 7,348 | 1 | 100–300 px |
| [Magnetic Tile](https://github.com/abin24/Magnetic-tile-defect-datasets) | ~8,642 | 1 | 50–150 px |

54 native defect labels → **8 unified semantic classes**: `contamination`, `cut`, `deformation`, `fracture`, `hole_void`, `minor_defect`, `scratch`, `surface_quality`

17 product categories across all three sources. Split: 70% train / 20% val / 10% test (stratified).

### Downloading the datasets

```bash
# MVTec AD (requires license agreement)
# Download from: https://www.mvtec.com/company/research/datasets/mvtec-ad

# Casting Product
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('ravirajsinh45/real-life-industrial-dataset-of-casting-product')"

# Magnetic Tile
git clone https://github.com/abin24/Magnetic-tile-defect-datasets.git
```

After downloading, update the paths in `data/merged_dataset_metadata_augmented.csv` to point to your local copies. A path remapping script is provided in `code/utils/remap_paths.py`.

---

## Installation

```bash
# Clone
git clone https://github.com/error-makerr/Industrial-Anomaly-Detection.git
cd Industrial-Anomaly-Detection

# Environment (Python 3.10+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn pandas matplotlib seaborn pillow scipy

# Optional (for FLOPs counting and web app)
pip install thop gradio flask
```

### Hardware requirements

| Task | Minimum | Recommended |
|---|---|---|
| Inference only | Any GPU with 4GB VRAM | RTX 3060 / 4070 Ti |
| Training EdgeNetV4 | 8GB VRAM GPU | RTX 3060 12GB+ |
| Training all baselines | 8GB VRAM GPU | RTX 4070 Ti 12GB |

---

## Pretrained Weights

| Model | Val F1 | Download |
|---|---|---|
| EdgeNetV4 (best, epoch 124) | 0.9662 | [Google Drive](#) |
| EdgeNetV3 | 0.912 | [Google Drive](#) |
| EdgeNetV2 | 0.844 | [Google Drive](#) |
| ResNet-50 (3-head baseline) | 0.728 | [Google Drive](#) |
| EfficientNet-B0 (3-head baseline) | 0.713 | [Google Drive](#) |
| DenseNet-121 (3-head baseline) | 0.616 | [Google Drive](#) |

> Replace `#` with your actual Google Drive or HuggingFace links after uploading.

---

## Quick Start

### Inference on a single image

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = EdgeNetV4()  # defined in code/edgenetv4_good_error_solved.ipynb
ckpt = torch.load('models/V4/EdgeNet_V4_best.pth', map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().cuda()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = transform(Image.open('test_image.png').convert('RGB')).unsqueeze(0).cuda()

# Inference — one forward pass, three answers
binary_logits, defect_logits, product_logits = model(img)

is_defective = torch.sigmoid(binary_logits).item() > 0.5
defect_class = defect_logits.argmax(1).item()
product_class = product_logits.argmax(1).item()

DEFECT_NAMES = ['contamination','cut','deformation','fracture',
                'hole_void','minor_defect','scratch','surface_quality']
PRODUCT_NAMES = ['bottle','cable','capsule','carpet','casting_product',
                 'grid','hazelnut','leather','magnetic_tile','metal_nut',
                 'pill','screw','tile','toothbrush','transistor','wood','zipper']

print(f"Defective: {is_defective}")
print(f"Defect type: {DEFECT_NAMES[defect_class]}")
print(f"Product: {PRODUCT_NAMES[product_class]}")
```

### Training from scratch

```bash
# 1. Prepare dataset (download + remap paths)
python code/utils/remap_paths.py --csv data/merged_dataset_metadata_augmented.csv \
                                  --mvtec /path/to/mvtec \
                                  --casting /path/to/casting \
                                  --magnetic /path/to/magnetic

# 2. Train EdgeNetV4
# Open code/training_script.ipynb and run all cells
# Or use code/edgenetv4_good_error_solved.ipynb for the debugged version

# 3. Train baselines (for comparison table)
# Open code/baseline_models.ipynb and run all cells
```

---

## Project Structure

```
Industrial-Anomaly-Detection/
├── code/
│   ├── training_script.ipynb          # Main EdgeNetV4 training
│   ├── edgenetv4_good_error_solved.ipynb  # Debugged training + evaluation
│   ├── baseline_models.ipynb          # 4 pretrained baseline training
│   ├── ablation_study.ipynb           # Architecture + training ablation
│   ├── confusion.ipynb                # Confusion matrix generation
│   ├── cross_dataset_validation.ipynb # Per-source domain evaluation
│   ├── defence.ipynb                  # Defense presentation figures
│   ├── synthetic_Gen.ipynb            # LoRA-based synthetic augmentation
│   ├── lora.py                        # LoRA training script
│   ├── effb2.ipynb                    # EfficientNet-B2 backbone experiments
│   └── extend.ipynb                   # Extended analysis and visualisation
├── data/
│   ├── merged_dataset_metadata_augmented.csv  # Master dataset index
│   ├── normalization_stats.json               # ImageNet norm stats
│   ├── mvtec/          # ← download separately
│   ├── casting/         # ← download separately
│   ├── magnetic/        # ← download separately
│   └── synthetic/       # ← generated by synthetic_Gen.ipynb
├── models/
│   ├── V2/EdgeNet_V2_best.pth
│   ├── V3/EdgeNet_V3_best.pth
│   ├── V4/EdgeNet_V4_best.pth         # Best model: F1 = 0.9662
│   └── baselines/
│       ├── efficientnet_b0_best.pth
│       ├── resnet50_best.pth
│       ├── mobilenetv2_best.pth
│       └── densenet121_best.pth
├── results/
│   ├── figures/          # Training curves, comparison tables, CM plots
│   ├── gradcam/          # Grad-CAM++ saliency maps
│   ├── ablation/         # Ablation study outputs
│   └── cross_dataset/    # Per-source evaluation results
├── webapp/               # DefectVision deployment code
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Results

### Model Comparison (all on same 21,344-image dataset, same 3-head framework)

| Model | Params | Binary Acc | Defect F1 | Product Acc |
|---|---|---|---|---|
| DenseNet-121 | ~8M | 97.84% | 0.616 | 100.0% |
| EfficientNet-B0 | ~5.4M | 97.30% | 0.713 | 100.0% |
| ResNet-50 | ~25M | 98.08% | 0.728 | 100.0% |
| EdgeNetV2 | 3.89M | 98.38% | 0.844 | 98.1% |
| EdgeNetV3 | 5.40M | 99.2% | 0.912 | 100.0% |
| **EdgeNetV4** | **7.88M** | **99.3%** | **0.9662** | **100.0%** |

### Per-Class Performance (validation, epoch 124)

| Class | Precision | Recall | F1 | Samples |
|---|---|---|---|---|
| contamination | 0.971 | 1.000 | 0.985 | 899 |
| cut | 0.889 | 0.941 | 0.914 | 605 |
| deformation | 0.938 | 1.000 | 0.968 | 1,177 |
| fracture | 1.000 | 1.000 | 1.000 | 732 |
| hole_void | 0.964 | 0.981 | 0.972 | 520 |
| minor_defect | 0.951 | 0.929 | 0.940 | **196** |
| scratch | 1.000 | 0.966 | 0.982 | 1,054 |
| surface_quality | 0.984 | 0.952 | 0.968 | 539 |

### Ablation Study

| Component removed | F1 drop |
|---|---|
| Multi-scale head → f4 only | −0.062 |
| Inverse-freq focal loss | −0.045 |
| Warm restarts (SGDR) | −0.045 |
| EMA | −0.016 |
| CoordAttention (inference) | −0.000 |

### Cross-Domain Stability

| Source Dataset | Binary F1 |
|---|---|
| MVTec AD | 0.952 |
| Casting Product | 0.999 |
| Magnetic Tile | 0.987 |

---

## Web Demo — DefectVision

<p align="center">
  <img src="results/figures/defectvision_demo.png" width="85%" alt="DefectVision Web Interface"/>
</p>

A live web interface that accepts an inspection image and returns:
- Binary classification (defective / non-defective) with confidence score
- Defect type identification (8 classes) with confidence
- Product category (17 classes) with confidence
- **Grad-CAM++ saliency heatmap** showing where the model is looking

```bash
# Run locally
cd webapp
python app.py
# Open http://localhost:5000
```

---

## Extending This Work

This codebase is designed to be reusable. Here's how to adapt it:

### Use your own dataset
1. Organise your images into folders by class
2. Create a CSV matching the format of `merged_dataset_metadata_augmented.csv`
3. Update `DEFECT_CLASSES` and `PRODUCT_CLASSES` lists in the training notebook
4. Adjust the number of output neurons in each head

### Add a new task head
The three-head design is modular. To add a fourth head (e.g. defect severity):
1. Define the new head in the model class (copy the product head pattern)
2. Add the new loss term to the multi-task balancing
3. Add the new label column to the CSV

### Change the backbone
Replace the `timm` backbone instantiation with any `features_only=True` compatible model:
```python
self.backbone = timm.create_model('efficientnet_b3', pretrained=True,
                                   features_only=True, out_indices=(2,3,4))
```

### Export for edge deployment
```python
dummy = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(model, dummy, "edgenetv4.onnx", opset_version=13)
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{sufi2026edgenetv4,
  title     = {Computer Vision Based Deep Learning Approaches for Automated
               Visual Inspection and Defect Detection in Industrial Environments},
  author    = {Sufi, Joynab Hossain and Arka, Aninda Sarkar and Fahim, Arfan Ahmed
               and Tamim, Most. Sadia Sultana and Hasan, Md. Jobayer},
  year      = {2026},
  school    = {BRAC University},
  type      = {B.Sc. Thesis},
  department = {Department of Computer Science and Engineering},
}
```

---

## License

This project is released for academic and research purposes. The source datasets retain their original licenses:
- MVTec AD: [CC BY-NC-SA 4.0](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Casting Product: [CC0 Public Domain](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- Magnetic Tile: [MIT License](https://github.com/abin24/Magnetic-tile-defect-datasets)

---

## Acknowledgements

- **Supervisor**: Dewan Ziaul Karim, Senior Lecturer, BRAC University
- **Co-supervisor**: Mohammad Rakibul Hasan Mahin, Lecturer, BRAC University
- Built with [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), and [scikit-learn](https://scikit-learn.org/)
