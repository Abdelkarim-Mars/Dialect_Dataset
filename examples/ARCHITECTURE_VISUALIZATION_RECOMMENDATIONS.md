# Architecture Visualization Recommendations for Academic Papers

**Date:** 2025-11-09
**Purpose:** Comprehensive guide to regenerating and improving neural network architecture diagrams for publication

---

## 📊 Current Status Assessment

### ✅ Existing Figures (Verified)

Your `examples/` folder contains **4 high-quality PNG files**:

| Figure | Resolution | Size | Status | Purpose |
|--------|-----------|------|--------|---------|
| `variant_a_complete.png` | 21625 × 1983 | 1.3 MB | ✅ Valid | Complete dual-branch architecture (main technical figure) |
| `variant_b_simplified.png` | 8970 × 5370 | 685 KB | ✅ Valid | High-level simplified pipeline (introduction/abstract) |
| `variant_c_preprocessing.png` | 5970 × 8970 | 1.0 MB | ✅ Valid | MFCC preprocessing pipeline (reproducibility) |
| `variant_d_training.png` | 7770 × 10170 | 1.5 MB | ✅ Valid | Two-stage training procedure (methodology) |

**Quality Assessment:**
- ✅ All files are valid PNG RGBA 8-bit images
- ✅ High resolution (300+ DPI equivalent)
- ✅ Publication-ready file sizes (<2MB each)
- ✅ Comprehensive documentation in markdown files
- ✅ Ready for journal submission

### 📚 Existing Documentation

| File | Content | Completeness |
|------|---------|-------------|
| `PNG_GENERATION_SUMMARY.md` | Complete summary with captions, alt-texts, usage notes | ⭐⭐⭐⭐⭐ |
| `architecture_diagrams_complete.md` | Detailed methodology extraction with 24 components | ⭐⭐⭐⭐⭐ |
| `methodology_extraction.md` | Complete algorithm with PlotNeuralNet LaTeX code | ⭐⭐⭐⭐⭐ |
| `export_subset.py` | Dataset utility script | ⭐⭐⭐⭐⭐ |
| `query_dataset.py` | Dataset query utility | ⭐⭐⭐⭐⭐ |

**Verdict:** Your existing figures are **excellent and publication-ready**. However, if you want to regenerate or improve them with more control, continue reading.

---

## 🎨 Best Python Libraries for Architecture Diagrams (2024-2025)

### **Tier 1: Publication-Quality Diagram Generators**

#### 1. **PlotNeuralNet** ⭐⭐⭐⭐⭐ (RECOMMENDED)
**Repository:** https://github.com/HarisIqbal88/PlotNeuralNet

**Why it's best for academic papers:**
- ✅ LaTeX-based: produces publication-quality vector graphics (PDF, EPS)
- ✅ Used in top-tier conference papers (CVPR, NeurIPS, ICLR)
- ✅ Highly customizable layer representations
- ✅ Professional 3D block-style diagrams
- ✅ Built-in templates for popular architectures

**Pros:**
- Best visual quality for academic papers
- Full control over layout, colors, dimensions
- Supports complex architectures (parallel branches, skip connections)
- Outputs vector formats (scalable without quality loss)
- Your methodology already uses this (LaTeX code provided)

**Cons:**
- Requires LaTeX installation
- Learning curve for LaTeX/TikZ
- Manual positioning of components
- Not interactive

**Installation:**
```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
cd PlotNeuralNet/pycore
# Ensure LaTeX installed: sudo apt-get install texlive-full
```

**Example Code (for your hybrid architecture):**
```python
import sys
sys.path.append('PlotNeuralNet/pycore/')
from pycore.tikzeng import *

# Create architecture file
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input layer
    to_input('input.png'),

    # CNN-LSTM Branch (Bottom)
    to_Conv("conv1", 256, 120, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="Conv1D-32"),
    to_Conv("conv2", 256, 60, offset="(2,0,0)", to="(conv1-east)", height=40, depth=40, width=2, caption="Conv1D-64"),
    to_Pool("pool1", offset="(0,0,0)", to="(conv2-east)", width=1, height=32, depth=32, opacity=0.5),
    to_Conv("conv3", 256, 60, offset="(2,0,0)", to="(pool1-east)", height=40, depth=40, width=3, caption="Conv1D-128"),
    to_Conv("conv4", 256, 60, offset="(2,0,0)", to="(conv3-east)", height=40, depth=40, width=4, caption="Conv1D-256"),

    # BiLSTM layers
    to_SoftMax("bilstm1", 256, "(2,0,0)", "(conv4-east)", caption="BiLSTM-1", width=4, height=40, depth=40),
    to_SoftMax("bilstm2", 256, "(2,0,0)", "(bilstm1-east)", caption="BiLSTM-2", width=4, height=40, depth=40),

    # Attention pooling
    to_SoftMax("attn", 256, "(2,0,0)", "(bilstm2-east)", caption="Attention Pool", width=2, height=30, depth=30),

    # Transformer Branch (Top) - parallel to CNN-LSTM
    to_Conv("wav2vec", 768, 120, offset="(0,6,0)", to="(input-east)", height=50, depth=50, width=5, caption="Wav2Vec 2.0"),
    to_ConvRelu("posenc", 768, 120, offset="(2,0,0)", to="(wav2vec-east)", caption="Pos Encoding"),
    to_Conv("transformer", 768, 120, offset="(3,0,0)", to="(posenc-east)", height=60, depth=60, width=8, caption="12× Transformer"),
    to_Pool("avgpool", offset="(2,0,0)", to="(transformer-east)", width=2, height=40, depth=40, caption="Avg Pool"),

    # Fusion
    to_ConvRelu("fusion", 1024, 1, offset="(3,0,0)", to="(avgpool-east)", caption="Concatenate"),

    # Classification head
    to_Conv("fc1", 512, 1, offset="(2,0,0)", to="(fusion-east)", caption="FC-512"),
    to_Conv("fc2", 256, 1, offset="(2,0,0)", to="(fc1-east)", caption="FC-256"),
    to_SoftMax("softmax", 20, "(2,0,0)", "(fc2-east)", caption="Softmax-20"),

    # Connections
    to_connection("input", "conv1"),
    to_connection("conv1", "conv2"),
    to_connection("conv2", "pool1"),
    to_connection("conv4", "bilstm1"),
    to_connection("bilstm2", "attn"),
    to_connection("wav2vec", "posenc"),
    to_connection("transformer", "avgpool"),
    to_connection("attn", "fusion"),
    to_connection("avgpool", "fusion"),
    to_connection("fc1", "fc2"),
    to_connection("fc2", "softmax"),

    to_end()
]

# Write to file
with open('hybrid_architecture.tex', 'w') as f:
    for line in arch:
        f.write(line)

# Compile: pdflatex hybrid_architecture.tex
```

---

#### 2. **VisualKeras** ⭐⭐⭐⭐
**Repository:** https://github.com/paulgavrikov/visualkeras

**Best for:** Quick, professional-looking diagrams with minimal code

**Pros:**
- ✅ Simple Python API (no LaTeX required)
- ✅ Automatically infers layer dimensions from Keras models
- ✅ Multiple visualization styles (layered, graph)
- ✅ Good for CNNs and sequential architectures
- ✅ Exports to PNG, SVG

**Cons:**
- ⚠️ Limited to Keras models
- ⚠️ Less control over styling than PlotNeuralNet
- ⚠️ Doesn't handle complex parallel branches as elegantly

**Installation:**
```bash
pip install visualkeras
```

**Example Code:**
```python
import visualkeras
from tensorflow import keras
from PIL import ImageFont

# Define your model
model = keras.Sequential([
    keras.layers.Conv1D(32, 7, input_shape=(120, 39)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv1D(64, 3),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(128, 3),
    keras.layers.Conv1D(256, 3),
    keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(128)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(20, activation='softmax')
])

# Generate visualization
font = ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(model,
                         legend=True,
                         font=font,
                         spacing=50,
                         to_file='architecture.png',
                         scale_xy=1,
                         scale_z=1,
                         max_z=1000)
```

---

#### 3. **Graphviz + Python (DOT language)** ⭐⭐⭐⭐
**Website:** https://graphviz.org/
**Python binding:** https://github.com/xflr6/graphviz

**Best for:** Maximum flexibility, node-edge diagrams

**Pros:**
- ✅ Full control over every aspect of the diagram
- ✅ Automatic layout algorithms (dot, neato, circo)
- ✅ Exports to many formats (PNG, SVG, PDF, EPS)
- ✅ Works for any architecture type
- ✅ Your Variant A already uses this approach

**Cons:**
- ⚠️ Requires manual specification of all nodes/edges
- ⚠️ Layout can be tricky for complex graphs
- ⚠️ Not as visually polished as PlotNeuralNet "out of the box"

**Installation:**
```bash
sudo apt-get install graphviz
pip install graphviz
```

**Example Code:**
```python
from graphviz import Digraph

dot = Digraph(comment='Hybrid CNN-LSTM-Transformer', format='png')
dot.attr(rankdir='LR', splines='ortho', nodesep='0.8', ranksep='1.2')
dot.attr('node', shape='box', style='filled,rounded', fontname='Arial')

# Input
dot.node('input', 'Raw Audio\n44.1 kHz', fillcolor='gray90')
dot.node('mfcc', 'MFCC Features\n120×39', fillcolor='gray80')

# CNN-LSTM Branch
dot.node('cnn1', 'Conv1D-32\nkernel=7', fillcolor='lightblue')
dot.node('cnn2', 'Conv1D-64\n+ MaxPool', fillcolor='lightblue')
dot.node('cnn3', 'Conv1D-128', fillcolor='lightblue')
dot.node('cnn4', 'Conv1D-256', fillcolor='lightblue')
dot.node('bilstm1', 'BiLSTM-1\nhidden=128×2', fillcolor='cornflowerblue')
dot.node('bilstm2', 'BiLSTM-2\nhidden=128×2', fillcolor='cornflowerblue')
dot.node('attn', 'Attention Pool\n→ 256-dim', fillcolor='lightsteelblue')

# Transformer Branch
dot.node('wav2vec', 'Wav2Vec 2.0\n7-layer CNN\n31.2M params', fillcolor='lightgreen')
dot.node('posenc', 'Positional\nEncoding', fillcolor='palegreen')
dot.node('transformer', '12× Transformer\n12 heads, d=768\n58.3M params', fillcolor='mediumseagreen')
dot.node('avgpool', 'Global Avg Pool\n→ 768-dim', fillcolor='lightgreen')

# Fusion & Classifier
dot.node('fusion', 'Concatenate\n1024-dim', fillcolor='orange')
dot.node('fc1', 'FC 1024→512\nDropout=0.5', fillcolor='lightsalmon')
dot.node('fc2', 'FC 512→256\nDropout=0.5', fillcolor='lightsalmon')
dot.node('output', 'Softmax\n20 classes', fillcolor='tomato')

# Edges
dot.edge('input', 'wav2vec')
dot.edge('input', 'mfcc')
dot.edge('mfcc', 'cnn1')
dot.edge('cnn1', 'cnn2')
dot.edge('cnn2', 'cnn3')
dot.edge('cnn3', 'cnn4')
dot.edge('cnn4', 'bilstm1')
dot.edge('bilstm1', 'bilstm2')
dot.edge('bilstm2', 'attn')
dot.edge('wav2vec', 'posenc')
dot.edge('posenc', 'transformer')
dot.edge('transformer', 'avgpool')
dot.edge('attn', 'fusion')
dot.edge('avgpool', 'fusion')
dot.edge('fusion', 'fc1')
dot.edge('fc1', 'fc2')
dot.edge('fc2', 'output')

# Render
dot.render('hybrid_architecture', view=False)
```

---

### **Tier 2: Quick & Interactive Tools**

#### 4. **Matplotlib + Custom Drawing** ⭐⭐⭐
**Best for:** Full Python control, programmatic generation

**Pros:**
- ✅ 100% Python (no external dependencies)
- ✅ Extremely flexible
- ✅ Can create any custom visualization
- ✅ Easy to integrate with data/experiments

**Cons:**
- ⚠️ Requires more code for complex diagrams
- ⚠️ Manual positioning of all elements
- ⚠️ Not as polished as specialized tools

**Example Code:**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(20, 8))
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis('off')

# Helper function to draw boxes
def draw_box(ax, x, y, w, h, text, color='lightblue'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=10, weight='bold')

# Draw architecture
draw_box(ax, 0, 3, 1.5, 1, 'Input\nAudio', 'gray')
draw_box(ax, 2, 3, 1.5, 1, 'MFCC\n120×39', 'lightgray')

# CNN-LSTM branch
draw_box(ax, 4, 2, 1.2, 0.8, 'Conv1D-32', 'lightblue')
draw_box(ax, 5.5, 2, 1.2, 0.8, 'Conv1D-64', 'lightblue')
draw_box(ax, 7, 2, 1.2, 0.8, 'Conv1D-128', 'lightblue')
draw_box(ax, 8.5, 2, 1.2, 0.8, 'Conv1D-256', 'lightblue')
draw_box(ax, 10, 2, 1.5, 0.8, 'BiLSTM', 'cornflowerblue')
draw_box(ax, 12, 2, 1.5, 0.8, 'Attention', 'steelblue')

# Transformer branch
draw_box(ax, 4, 5, 2, 1, 'Wav2Vec 2.0', 'lightgreen')
draw_box(ax, 6.5, 5, 1.5, 1, 'Pos Enc', 'palegreen')
draw_box(ax, 8.5, 5, 2.5, 1, '12× Transformer', 'mediumseagreen')
draw_box(ax, 11.5, 5, 1.5, 1, 'Avg Pool', 'lightgreen')

# Fusion
draw_box(ax, 14, 3.5, 1.5, 1, 'Concat\n1024', 'orange')

# Classifier
draw_box(ax, 16, 3.5, 1.2, 0.8, 'FC-512', 'lightsalmon')
draw_box(ax, 17.5, 3.5, 1.2, 0.8, 'FC-256', 'lightsalmon')
draw_box(ax, 19, 3.5, 1, 0.8, 'Softmax\n20', 'tomato')

# Add arrows (simplified)
arrow_props = dict(arrowstyle='->', lw=2, color='black')
ax.annotate('', xy=(2, 3.5), xytext=(1.5, 3.5), arrowprops=arrow_props)
ax.annotate('', xy=(4, 2.4), xytext=(3.5, 3.3), arrowprops=arrow_props)
# ... add more arrows

plt.title('Hybrid CNN-LSTM-Transformer Architecture', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('architecture_matplotlib.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

#### 5. **NN-SVG** ⭐⭐⭐
**Website:** http://alexlenail.me/NN-SVG/

**Best for:** Quick web-based visualization without coding

**Pros:**
- ✅ No installation required (web-based)
- ✅ Interactive GUI
- ✅ Exports to SVG
- ✅ Templates for common architectures (LeNet, AlexNet, etc.)

**Cons:**
- ⚠️ Limited customization
- ⚠️ Not programmatic (manual editing)
- ⚠️ Doesn't handle complex parallel branches well

**Usage:** Visit the website, use the interactive interface to design your architecture, download SVG.

---

#### 6. **draw_convnet** ⭐⭐⭐
**Repository:** https://github.com/gwding/draw_convnet

**Best for:** Python-based CNN visualization with minimal code

**Pros:**
- ✅ Pure Python
- ✅ Easy to use for CNNs
- ✅ Exports to PNG

**Cons:**
- ⚠️ Limited to CNNs
- ⚠️ Not actively maintained
- ⚠️ Less flexible than other options

---

### **Tier 3: Specialized Tools**

#### 7. **Netron** (Model Viewer)
**Website:** https://netron.app/

**Best for:** Viewing existing trained models

**Pros:**
- ✅ Supports all major frameworks (PyTorch, TensorFlow, ONNX)
- ✅ Interactive exploration
- ✅ Shows tensor shapes and parameters

**Cons:**
- ⚠️ Not for creating publication diagrams
- ⚠️ Viewer only (not a diagram creator)

---

#### 8. **TensorBoard** (Training Visualization)
**Installation:** `pip install tensorboard`

**Best for:** Interactive model exploration during training

**Pros:**
- ✅ Built into TensorFlow
- ✅ Real-time training metrics
- ✅ Computational graph visualization

**Cons:**
- ⚠️ Not suitable for static publication figures
- ⚠️ Requires TensorFlow

---

## 🎯 Recommendations for Your Specific Case

### Your Architecture: Hybrid CNN-LSTM-Transformer

Based on your existing figures and documentation, here are my recommendations:

### **Option 1: Keep Current Figures (RECOMMENDED)** ✅

**Verdict:** Your existing figures are **excellent quality** and ready for publication.

**Reasons:**
1. ✅ High resolution (>300 DPI equivalent)
2. ✅ Clear visual hierarchy (color-coded branches)
3. ✅ Comprehensive (4 variants for different purposes)
4. ✅ Professional documentation
5. ✅ Already validated against methodology

**Action:** Use current figures as-is for paper submission.

---

### **Option 2: Enhance with PlotNeuralNet** (IF YOU WANT MORE CONTROL)

**When to use:**
- You want 3D block-style diagrams like those in CVPR/NeurIPS papers
- You need vector output (PDF/EPS) for journal requirements
- You want to adjust visual style (colors, spacing, annotations)

**Steps:**
1. Use the LaTeX code from `methodology_extraction.md` (lines 676-863)
2. Customize layer appearance (change colors, sizes, annotations)
3. Compile: `pdflatex hybrid_architecture.tex`
4. Convert to PNG if needed: `pdftocairo -png -r 300 hybrid_architecture.pdf`

**Time estimate:** 2-4 hours (including LaTeX setup and tweaking)

---

### **Option 3: Create Python-Native Version with Matplotlib**

**When to use:**
- You want to programmatically generate diagrams
- You need to create variations for different experiments
- You prefer pure Python (no LaTeX dependencies)

**Steps:**
1. Write Python script using Matplotlib (see Tier 2, example above)
2. Generate figures programmatically
3. Export to PNG/SVG

**Time estimate:** 3-5 hours (custom drawing code)

---

### **Option 4: Quick Update with Graphviz**

**When to use:**
- You want to update Variant A with minor changes
- You need to add/remove components
- You want automatic layout

**Steps:**
1. Modify the DOT source code (likely used to generate Variant A)
2. Regenerate with Graphviz: `dot -Tpng -Gdpi=300 architecture.dot -o output.png`

**Time estimate:** 1-2 hours

---

## 📋 Step-by-Step Guide: Regenerating Figures

### Method 1: PlotNeuralNet (Recommended for New Figures)

```bash
# 1. Install dependencies
sudo apt-get update
sudo apt-get install texlive-full graphviz
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git

# 2. Use the provided LaTeX template from your methodology_extraction.md
cd PlotNeuralNet/pycore
# Copy the LaTeX code from examples/methodology_extraction.md (lines 676-863)
# Save as hybrid_architecture.tex

# 3. Compile
pdflatex hybrid_architecture.tex

# 4. Convert to PNG if needed
pdftocairo -png -r 300 hybrid_architecture.pdf hybrid_architecture.png

# 5. Optimize
optipng hybrid_architecture-1.png  # Reduce file size
```

---

### Method 2: Python + Matplotlib

```python
# Complete script for generating architecture diagram

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

# Configuration
fig = plt.figure(figsize=(24, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 24)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
colors = {
    'input': '#E0E0E0',
    'cnn': '#90CAF9',
    'lstm': '#64B5F6',
    'transformer': '#81C784',
    'fusion': '#FFB74D',
    'classifier': '#FF8A65',
    'output': '#E57373'
}

# Helper function
def add_box(x, y, width, height, text, color, ax):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=9, weight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))

# Add layers (simplified example - expand as needed)
add_box(0, 4, 1.5, 1.5, 'Input\nAudio\n44.1kHz', colors['input'], ax)
add_box(2, 4, 1.5, 1.5, 'MFCC\n120×39', colors['input'], ax)

# CNN-LSTM branch
cnn_y = 2
add_box(4, cnn_y, 1.2, 1, 'Conv1D\n32', colors['cnn'], ax)
add_box(5.5, cnn_y, 1.2, 1, 'Conv1D\n64', colors['cnn'], ax)
add_box(7, cnn_y, 1.2, 1, 'Conv1D\n128', colors['cnn'], ax)
add_box(8.5, cnn_y, 1.2, 1, 'Conv1D\n256', colors['cnn'], ax)
add_box(10.5, cnn_y, 1.5, 1, 'BiLSTM\n2 layers', colors['lstm'], ax)
add_box(12.5, cnn_y, 1.5, 1, 'Attention\nPool', colors['lstm'], ax)

# Transformer branch
trans_y = 6
add_box(4, trans_y, 2, 1.2, 'Wav2Vec 2.0\n31.2M', colors['transformer'], ax)
add_box(6.5, trans_y, 1.5, 1.2, 'Pos Enc', colors['transformer'], ax)
add_box(8.5, trans_y, 3, 1.2, '12× Transformer\n58.3M', colors['transformer'], ax)
add_box(12, trans_y, 1.5, 1.2, 'Avg Pool', colors['transformer'], ax)

# Fusion
add_box(15, 4.5, 1.8, 1.2, 'Concat\n1024-dim', colors['fusion'], ax)

# Classifier
add_box(17.5, 4.5, 1.5, 1, 'FC 512', colors['classifier'], ax)
add_box(19.5, 4.5, 1.5, 1, 'FC 256', colors['classifier'], ax)
add_box(21.5, 4.5, 1.5, 1, 'Softmax\n20', colors['output'], ax)

# Add title
plt.title('Hybrid CNN-LSTM-Transformer Architecture for Arabic Word Recognition',
         fontsize=18, weight='bold', pad=20)

# Add legend
legend_elements = [
    mpatches.Patch(color=colors['cnn'], label='CNN Layers'),
    mpatches.Patch(color=colors['lstm'], label='LSTM + Attention'),
    mpatches.Patch(color=colors['transformer'], label='Transformer'),
    mpatches.Patch(color=colors['fusion'], label='Fusion'),
    mpatches.Patch(color=colors['classifier'], label='Classifier'),
    mpatches.Patch(color=colors['output'], label='Output')
]
ax.legend(handles=legend_elements, loc='lower center',
         bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=12)

# Save
plt.tight_layout()
plt.savefig('hybrid_architecture_matplotlib.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✓ Figure saved: hybrid_architecture_matplotlib.png")
```

---

### Method 3: Graphviz (Quick Updates)

```bash
# Install
sudo apt-get install graphviz

# Create DOT file (architecture.dot)
cat > architecture.dot << 'EOF'
digraph Architecture {
    rankdir=LR;
    splines=ortho;
    nodesep=0.8;
    ranksep=1.5;

    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=12];

    // Input
    input [label="Input\nAudio\n44.1kHz", fillcolor="gray90"];
    mfcc [label="MFCC\n120×39", fillcolor="gray80"];

    // CNN-LSTM branch
    cnn1 [label="Conv1D-32\nk=7", fillcolor="lightblue"];
    cnn2 [label="Conv1D-64\n+ Pool", fillcolor="lightblue"];
    cnn3 [label="Conv1D-128", fillcolor="lightblue"];
    cnn4 [label="Conv1D-256", fillcolor="lightblue"];
    bilstm [label="2× BiLSTM\nhidden=128", fillcolor="cornflowerblue"];
    attn [label="Attention\n→ 256", fillcolor="steelblue"];

    // Transformer branch
    wav2vec [label="Wav2Vec 2.0\n31.2M params", fillcolor="lightgreen"];
    posenc [label="Positional\nEncoding", fillcolor="palegreen"];
    transformer [label="12× Transformer\n12 heads, d=768\n58.3M params", fillcolor="mediumseagreen"];
    avgpool [label="Global\nAvg Pool\n→ 768", fillcolor="lightgreen"];

    // Fusion
    fusion [label="Concatenate\n1024-dim", fillcolor="orange"];

    // Classifier
    fc1 [label="FC 1024→512\nDropout=0.5", fillcolor="lightsalmon"];
    fc2 [label="FC 512→256\nDropout=0.5", fillcolor="lightsalmon"];
    output [label="Softmax\n20 classes", fillcolor="tomato"];

    // Edges
    input -> {mfcc, wav2vec};
    mfcc -> cnn1 -> cnn2 -> cnn3 -> cnn4 -> bilstm -> attn;
    wav2vec -> posenc -> transformer -> avgpool;
    {attn, avgpool} -> fusion;
    fusion -> fc1 -> fc2 -> output;
}
EOF

# Generate PNG
dot -Tpng -Gdpi=300 architecture.dot -o architecture.png

# Generate PDF (vector)
dot -Tpdf architecture.dot -o architecture.pdf

# Generate SVG
dot -Tsvg architecture.dot -o architecture.svg
```

---

## 🚀 Quick Start Recommendation

### For Immediate Use (TODAY):

**Use your existing figures** - they're already publication-ready!

### For Future Improvements (THIS WEEK):

1. **Try PlotNeuralNet** for 3D block-style variants
2. **Experiment with Graphviz** for quick layout iterations
3. **Keep Matplotlib** as backup for custom modifications

### Installation Command (One-Line):

```bash
# Install all recommended tools
sudo apt-get update && sudo apt-get install -y texlive-full graphviz && \
pip install visualkeras graphviz matplotlib pillow && \
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

---

## 📚 Additional Resources

### Tutorials
1. **PlotNeuralNet Tutorial:** https://towardsai.net/p/l/creating-stunning-neural-network-visualizations-with-chatgpt-and-plotneuralnet
2. **Graphviz Gallery:** https://graphviz.org/gallery/
3. **Matplotlib Architecture Examples:** https://matplotlib.org/stable/gallery/index.html

### Example Papers with Great Figures
1. **U-Net:** https://arxiv.org/abs/1505.04597 (classic architecture figure)
2. **ResNet:** https://arxiv.org/abs/1512.03385 (block diagrams)
3. **Transformer:** https://arxiv.org/abs/1706.03762 (attention visualization)

### Tools Comparison Chart

| Tool | Quality | Ease of Use | Flexibility | Learning Curve | Output Format |
|------|---------|-------------|-------------|----------------|---------------|
| **PlotNeuralNet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium-High | PDF, PNG |
| **VisualKeras** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Low | PNG, SVG |
| **Graphviz** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low-Medium | PNG, PDF, SVG |
| **Matplotlib** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | PNG, PDF, SVG |
| **NN-SVG** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Very Low | SVG |

---

## ✅ Final Verdict

### **Your Current Situation:**
✅ You already have **4 excellent publication-ready figures**
✅ Comprehensive documentation exists
✅ High resolution and professional quality

### **My Recommendation:**

**PRIMARY:** Use your existing figures as-is for paper submission.

**SECONDARY:** If you want to experiment with different visual styles:
1. **For 3D block diagrams:** Try PlotNeuralNet
2. **For quick iterations:** Use Graphviz
3. **For full Python control:** Use Matplotlib

### **Installation Priority:**
```bash
# Essential (if you want to regenerate)
sudo apt-get install graphviz texlive-full

# Nice to have
pip install visualkeras graphviz matplotlib

# Optional (for experiments)
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

---

## 📞 Need Help?

If you want me to:
- Generate new figures with specific style
- Modify existing figures
- Create custom Python scripts
- Convert between formats

Just ask! I can help you create publication-quality architecture diagrams tailored to your needs.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Status:** ✅ Complete and Ready to Use
