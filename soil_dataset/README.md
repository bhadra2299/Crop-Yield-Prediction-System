# IntelliCrop – Soil Image Dataset
**4,200 synthetic soil images · 10 classes · 224×224 px · JPEG**

## Classes & Distribution
| Class         | Count | Texture Style     | Key Features                    |
|---------------|-------|-------------------|---------------------------------|
| Alluvial Soil |  420  | Layered           | Striated, pebbles, roots        |
| Black Soil    |  420  | Fine Clay         | Dark, crack patterns, high moisture |
| Chalky Soil   |  420  | Powdery           | Light/white, limestone pebbles  |
| Clay Soil     |  420  | Smooth Clay       | Cracks when dry, high cohesion  |
| Laterite Soil |  420  | Rough Granular    | Orange-red, iron-rich, coarse   |
| Loamy Soil    |  420  | Mixed Grain       | Brown, organic matter, roots    |
| Peat Soil     |  420  | Fibrous Organic   | Dark brown, root fibers, moist  |
| Red Soil      |  420  | Granular          | Iron oxide red, coarse grain    |
| Sandy Soil    |  420  | Coarse Grain      | Light tan, individual grains    |
| Silt Soil     |  420  | Fine Smooth       | Grey-brown, fine particles      |

## Image Properties
- **Resolution:** 224 × 224 pixels (ImageNet standard)
- **Format:** JPEG (quality 92)
- **Total size:** ~20 MB
- **Color space:** RGB

## Generation Technique
Each image uses procedural texture synthesis:
- Multi-octave fractal Perlin noise (5–6 octaves)
- Voronoi cell texture for granular soils
- Desiccation crack simulation (black/clay)
- Root fiber rendering (peat/loamy)
- Random pebble/stone inclusions
- Moisture-based color darkening
- Brightness, contrast, saturation jitter
- Gaussian blur + unsharp mask
- Slight rotation augmentation

## Usage with PyTorch / TensorFlow

### TensorFlow / Keras
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    'soil_dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
```

### PyTorch
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
dataset = datasets.ImageFolder('soil_dataset/', transform=transform)
```

## Re-generating
```bash
python generate_soil_images.py
```
Modify `IMAGES_PER_CLASS` and `IMG_SIZE` in the script for custom output.

---
**Author:** Kudum Veerabhadraiah · IntelliCrop AI Agriculture System
