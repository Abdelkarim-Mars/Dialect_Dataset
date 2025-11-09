# ASR Methodology Extraction: Hybrid CNN-LSTM-Transformer for Arabic Isolated Word Recognition

## 1. Assumptions & Scope

**None required** — The methodology section provides comprehensive detail on all aspects:
- Complete dataset construction protocol (50 speakers, 9,992 utterances, 20 classes)
- Detailed MFCC extraction pipeline (8 steps with all parameters)
- Full architecture specification (layer-wise equations, dimensions, parameters)
- Exhaustive training procedures (two-stage protocol, hyperparameters, augmentations)
- Comprehensive evaluation metrics and protocols

**Task Clarification**: This is an **isolated word classification task** (20 Arabic words/numerals), NOT continuous speech recognition. The "ASR" terminology in the prompt template refers more broadly to speech processing. The model outputs categorical predictions via softmax, not transcriptions via CTC/AED/RNN-T.

---

## 2. Method Summary

### 2.1 Task Formulation

- **Task Type**: Isolated word classification (supervised learning)
- **Input**: Single-word utterances (200–1500 ms duration)
- **Output**: Class label $\in \{0, 1, \ldots, 19\}$ representing Arabic numerals (0–9) and command words
- **Formulation**: Multi-class classification with 20 balanced classes
- **No decoder**: Direct classification via fully connected layers + softmax
- **No language model**: Task does not require linguistic context beyond single words
- **Architecture**: Parallel dual-branch hybrid model (CNN-LSTM + Transformer)

### 2.2 Data & Features

#### Dataset: Arabic Isolated Words and Numerals
- **Source**: Custom corpus (Section 3.1)
- **Speakers**: 50 native Arabic speakers (male, aged 22–45, Tunisian dialect)
- **Vocabulary**: 20 words (numerals 0–9 + 10 command words)
- **Total samples**: 9,992 utterances
- **Recording specs**:
  - Sampling rate: **44,100 Hz** (broadcast quality)
  - Bit depth: 16-bit linear PCM
  - Format: Uncompressed WAV
  - Environment: Quiet indoor (<30 dB SPL ambient noise)
  - Microphone: Audio-Technica AT2020 cardioid condenser (15–20 cm distance)
- **Duration**: Mean 563 ms (range: 180–980 ms)
- **Quality control**: Automatic checks (clipping, SNR >20 dB, silence trimming) + manual verification (2 annotators)

#### Data Partitioning (Speaker-Disjoint Split)
- **Training**: 6,494 utterances (65%) from 33 speakers
- **Validation**: 998 utterances (10%) from 5 speakers  
- **Test**: 2,500 utterances (25%) from 12 speakers
- **Critical constraint**: No speaker overlap across sets (strict speaker-independent evaluation)

#### Feature Extraction: MFCCs (Section 3.2)

**Pipeline** (8 sequential steps via `librosa 0.9.2`):

1. **Pre-emphasis**: $y[n] = x[n] - 0.97 \cdot x[n-1]$
2. **Framing**: 
   - Frame length: 25 ms (1,103 samples at 44.1 kHz)
   - Hop length: 10 ms (441 samples, 75% overlap)
   - Yields ~56 frames for average utterance (563 ms)
3. **Windowing**: Hann window (tapers to zero at boundaries)
4. **STFT**: $N_{\text{FFT}} = 2048$, magnitude spectrum $|X(m, k)|$
5. **Mel-filterbank**: 40 triangular filters, Mel scale: $\text{mel}(f) = 2595 \log_{10}(1 + f/700)$
6. **Log compression**: $S(m, i) = \log(E(m, i) + 10^{-10})$
7. **DCT**: Extract first 13 cepstral coefficients $c_1, \ldots, c_{13}$ (discard $c_0$)
8. **Delta features**: Compute $\Delta$ and $\Delta\Delta$ via 5-frame regression

**Final representation**: 
- **Dimension**: 39 (13 static + 13 delta + 13 delta-delta)
- **Sequence length**: $T_{\max} = 120$ frames (1.2 sec, zero-padded or truncated)
- **Input tensor shape**: $(B, 120, 39)$

#### Feature Normalization
- **Method**: Speaker-independent global normalization
- $\hat{\mathbf{f}}(m) = (\mathbf{f}(m) - \boldsymbol{\mu}_{\text{train}}) / \boldsymbol{\sigma}_{\text{train}}$
- Statistics ($\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}}$) computed on training set, applied to all sets

### 2.3 Model Architecture

**Key Innovation**: **Parallel dual-branch design** (not sequential stacking)
- **Branch 1 (CNN-LSTM)**: Local acoustic features + short-term temporal dependencies
- **Branch 2 (Transformer)**: Global context + long-range dependencies via self-attention
- **Advantages**: Preserves gradient flow, enables specialization, provides robustness via fusion

#### Branch 1: CNN-LSTM for Local Features (Section 3.3.1)

**Convolutional Layers** (4× 1D Conv along temporal axis):

```
Input: X ∈ ℝ^(B×120×39)

h₁ = ReLU(BN(Conv1D_{39→32}(X)))           [kernel=7, stride=1]
h₂ = MaxPool(ReLU(BN(Conv1D_{32→64}(h₁))))  [kernel=3; MaxPool: k=2, s=2 → 60 frames]
h₃ = ReLU(BN(Conv1D_{64→128}(h₂)))          [kernel=3]
h₄ = ReLU(BN(Conv1D_{128→256}(h₃)))         [kernel=3]

Output: h₄ ∈ ℝ^(B×60×256)
```

- **Design choices**: 1D conv (treats 39 MFCC dims as channels), increasing filters (32→64→128→256), BatchNorm for stability
- **Temporal downsampling**: Only after 2nd layer (120→60 frames via MaxPool)

**Bidirectional LSTM Layers** (2× stacked BiLSTM):

```
h₅ = BiLSTM^(1)(h₄)  ∈ ℝ^(B×60×256)  [hidden=128 per direction, concatenated]
h₆ = BiLSTM^(2)(h₅)  ∈ ℝ^(B×60×256)  [hidden=128 per direction]
```

- **Dropout**: 0.3 on recurrent connections (variational dropout)
- **LSTM gates**: Forget, input, output gates + cell state (standard LSTM equations)

**Attention-Based Temporal Pooling**:

```
u_t = tanh(W_a · h₆,t + b_a)           [W_a ∈ ℝ^(128×256)]
α_t = exp(u_t^T v) / Σ_t' exp(u_t'^T v)  [v ∈ ℝ^128 learned context]
z_CNN-LSTM = Σ_t α_t · h₆,t            [∈ ℝ^(B×256)]
```

- **Purpose**: Aggregate variable-length sequence → fixed-size representation
- **Interpretation**: α_t assigns importance to phonetically salient frames

#### Branch 2: Transformer with Wav2Vec 2.0 Pre-training (Section 3.3.2)

**Wav2Vec 2.0 Feature Extractor**:
- **Source**: Pre-trained on 960h LibriSpeech (English, self-supervised contrastive learning)
- **Architecture**: 7-layer CNN encoder with strides [5,2,2,2,2,2,2]
- **Input**: Raw waveform (44,100 samples for 1 sec)
- **Output**: $\mathbf{H}_{\text{wav2vec}} \in \mathbb{R}^{B \times 120 \times 768}$ (conveniently matches MFCC temporal dim)

**Transfer Learning Strategy** (Two-stage):
1. **Epochs 1–10**: Freeze Wav2Vec (31.2M params), train only Transformer encoder + classifier
2. **Epochs 11–50**: Unfreeze Wav2Vec, fine-tune with reduced LR ($5 \times 10^{-5}$, 20× lower)

**Positional Encoding** (Sinusoidal):

```
PE(t, 2i)   = sin(t / 10000^(2i/768))
PE(t, 2i+1) = cos(t / 10000^(2i/768))

H₀ = H_wav2vec + PE   [∈ ℝ^(B×120×768)]
```

**Transformer Encoder** (12 layers, each with):

**(a) Multi-Head Self-Attention**:
- Heads: $h = 12$
- Dimension per head: $d_k = d_v = 768/12 = 64$
- $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$
- $\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_{12}) W^O$

**(b) Layer Norm + Residual**: 
- $\tilde{H}_\ell = \text{LayerNorm}(H_{\ell-1} + \text{MultiHead}(H_{\ell-1}))$

**(c) Position-wise FFN**:
- $\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$
- Hidden dim: $4 \times d_{\text{model}} = 3072$

**(d) Second Residual**:
- $H_\ell = \text{LayerNorm}(\tilde{H}_\ell + \text{FFN}(\tilde{H}_\ell))$

**Global Average Pooling**:

```
z_Transformer = (1/120) Σ_t H₁₂,t   [∈ ℝ^(B×768)]
```

#### Fusion and Classification Head (Section 3.3.3)

**Feature Fusion** (Concatenation):

```
z_fused = [z_CNN-LSTM ; z_Transformer]   [∈ ℝ^(B×1024)]
```

- **Alternative strategies tested** (ablations): Learned attention (97.21%), element-wise product (94.12%)
- **Concatenation chosen**: Best performance (97.82%), preserves all information

**Classification Head** (2× FC layers + Dropout):

```
h₇ = Dropout(ReLU(BN(W₁·z_fused + b₁)))   [∈ ℝ^(B×512), dropout=0.5]
h₈ = Dropout(ReLU(BN(W₂·h₇ + b₂)))        [∈ ℝ^(B×256), dropout=0.5]
y  = Softmax(W₃·h₈ + b₃)                  [∈ ℝ^(B×20)]
```

**Loss Function**: Cross-entropy with label smoothing ($\epsilon = 0.1$):

```
L = -(1/B) Σ_b Σ_i ỹ_b,i log(ŷ_b,i)

ỹ_b,i = {1 - ε + ε/20   if i = true class
        {ε/20           otherwise
```

- **Purpose**: Prevents overconfident predictions, improves calibration

#### Model Complexity (Section 3.3.4)

| Component | Parameters | FLOPs | % Total |
|-----------|-----------|-------|---------|
| **CNN-LSTM Branch** | 1.2M | 2.8G | 1.3% |
| **Transformer Branch** | 89.5M | 145.2G | 97.6% |
| - Wav2Vec Feature Extractor | 31.2M | 48.5G | |
| - Transformer Encoder (12×) | 58.3M | 96.7G | |
| **Classification Head** | 1.0M | 0.3G | 1.1% |
| **Total (Hybrid)** | **91.7M** | **148.3G** | **100%** |

- **Inference latency**: 5.3 ms (NVIDIA V100, batch=1)
- **Model size**: 366 MB checkpoint (32-bit floats)

### 2.4 Training Procedure

#### Two-Stage Training Strategy (Section 3.4.1)

**Stage 1: Feature Extraction (Epochs 1–10)**
- **Frozen**: Wav2Vec 2.0 feature extractor (31.2M params)
- **Trainable**: CNN-LSTM, Transformer encoder (excl. Wav2Vec), classifier (60.5M params)
- **Learning rates**: $10^{-4}$ (CNN-LSTM), $2 \times 10^{-4}$ (Transformer encoder)

**Stage 2: Full Fine-Tuning (Epochs 11–50)**
- **Trainable**: All parameters (91.7M total)
- **Learning rates**: $10^{-4}$ (CNN-LSTM/classifier), $5 \times 10^{-5}$ (Wav2Vec, discriminative fine-tuning)

#### Optimization Configuration (Section 3.4.2)

**Optimizer**: Adam
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$
- Weight decay: $10^{-4}$ (L2 regularization)

**Learning Rate Schedule**: Cosine annealing with warm restarts
- $\eta_t = \eta_{\min} + 0.5(\eta_{\max} - \eta_{\min})(1 + \cos(T_{\text{cur}}/T_{\max} \pi))$
- $\eta_{\max} = 10^{-4}$, $\eta_{\min} = 10^{-6}$
- Cycle length: $T_{\max} = 10$ epochs
- Restarts after epochs: [10, 20, 30, 40]

**Batch Configuration**:
- Batch size: 32 (Arabic dataset)
- Gradient accumulation: 4 steps (effective batch = 128)
- Gradient clipping: Max norm 1.0 (prevents exploding gradients in RNNs)

**Early Stopping**:
- Monitor: Validation accuracy
- Patience: 10 epochs
- Restore best checkpoint on validation set

#### Data Augmentation (Section 3.4.3)

- **Time stretching**: Factor ~ U(0.9, 1.1)
- **Pitch shifting**: ±2 semitones ~ U(-2, 2)
- **Additive noise**: Gaussian N(0, σ²), σ = 0.005
- **SpecAugment** (50% probability):
  - Time masking: T = 20 consecutive frames
  - Frequency masking: F = 5 consecutive MFCC coefficients
- **Mixup** (α = 0.2):
  - $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, λ ~ Beta(0.2, 0.2)
  - $\tilde{y} = \lambda y_i + (1-\lambda) y_j$

#### Hyperparameter Optimization (Section 3.4.4)

**Method**: Bayesian optimization (50 trials, Expected Improvement acquisition, Gaussian Process surrogate)

**Key optimal values**:
- LR (CNN-LSTM): $10^{-4}$
- LR (Transformer): $5 \times 10^{-5}$
- Batch size: 32
- Dropout (FC): 0.5
- Dropout (LSTM): 0.3
- LSTM hidden: 128
- Transformer layers: 12
- Attention heads: 12
- Label smoothing: 0.1
- Weight decay: $10^{-4}$

### 2.5 Decoding

**Not applicable** — This is a classification task, not a sequence-to-sequence ASR task.

**Inference procedure**:
1. Extract 39-dim MFCCs from test utterance
2. Forward pass through both branches
3. Fuse representations via concatenation
4. Pass through classification head
5. Apply softmax to obtain class probabilities
6. Predict class with maximum probability: $\hat{y} = \arg\max_i p(y=i | x)$

### 2.6 Evaluation

#### Primary Metrics (Section 3.5.1)

- **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$
- **Per-class Precision**: $TP_c / (TP_c + FP_c)$
- **Per-class Recall**: $TP_c / (TP_c + FN_c)$
- **Per-class F1**: $2 \cdot (P_c \cdot R_c) / (P_c + R_c)$
- **Macro-F1**: $(1/20) \sum_{c=1}^{20} F1_c$ (equal weight per class)
- **Weighted-F1**: $\sum_{c=1}^{20} w_c \cdot F1_c$ (weighted by class frequency)

#### Confusion Matrix Analysis (Section 3.5.2)
- Matrix $C \in \mathbb{R}^{20 \times 20}$ where $C_{ij}$ = count(true=i, pred=j)
- Diagonal: Correct predictions
- Off-diagonal: Systematic confusions (often phonetically motivated)

#### Statistical Significance (Section 3.5.3)
- **McNemar's test**: Compare error rates between models ($p < 0.001$)
- **Paired t-test**: Compare performance across 5 random seeds

#### Calibration Metrics (Section 3.5.4)
- **Expected Calibration Error (ECE)**: Measures alignment between predicted probabilities and actual accuracy (M=10 bins)
- **Maximum Calibration Error (MCE)**: Max deviation across bins
- **Brier Score**: $(1/N) \sum_i \sum_c (\hat{y}_{i,c} - y_{i,c})^2$

#### Additional Evaluations
- **ROC curves**: Micro-averaged TPR vs. FPR
- **AUC**: Area under ROC curve (1.0 = perfect)
- **Precision-Recall curves**: More informative for balanced datasets
- **Inference latency**: Mean ± std over 1,000 samples (NVIDIA V100)
- **Memory footprint**: Peak GPU memory (MB)
- **Model size**: Parameters × 4 bytes / 1024² (MB, 32-bit float)

#### Robustness Evaluation (Section 3.5.5)
- **Noise robustness**: Inject additive white Gaussian noise at SNR ∈ {0, 5, 10, 15, 20, Clean} dB
- **Cross-dialectal generalization**: Test on Egyptian, Levantine, Gulf, Moroccan dialects (held-out speakers)

### 2.7 References (by section/label from methodology.tex)

- **Section 3.1**: Dataset construction (lines 9–149)
- **Section 3.2**: MFCC extraction (lines 171–307)
- **Section 3.3**: Model architecture (lines 308–619)
  - **Section 3.3.1**: CNN-LSTM branch (lines 333–430)
  - **Section 3.3.2**: Transformer branch (lines 431–524)
  - **Section 3.3.3**: Fusion and classification (lines 525–580)
  - **Section 3.3.4**: Model complexity (lines 581–619)
- **Section 3.4**: Training procedure (lines 620–749)
- **Section 3.5**: Evaluation metrics (lines 750–904)
- **Section 3.6**: Implementation details (lines 905–940)

---

## 3. Algorithm: End-to-End Training & Inference

```
ALGORITHM: HYBRID_CNN_LSTM_TRANSFORMER_ARABIC_WORD_CLASSIFICATION

================================================================================
INPUT: 
    D_train: Training dataset (6,494 Arabic utterances, 33 speakers)
    D_val:   Validation dataset (998 utterances, 5 speakers)
    D_test:  Test dataset (2,500 utterances, 12 speakers)
    Y:       Class labels ∈ {0, 1, ..., 19} for 20 Arabic words
    C:       Configuration dict (hyperparameters)

OUTPUT:
    θ*:      Trained model parameters (91.7M params)
    Ŷ_test:  Predicted labels for test set
    metrics: Accuracy, F1, confusion matrix, calibration scores

================================================================================
PREPROCESSING (Section 3.2):
    
    For each audio waveform x ∈ D:
        // MFCC Extraction (8-step pipeline)
        x ← resample(x, fs=44100)                      // Ensure consistent sampling rate
        y ← pre_emphasis(x, α=0.97)                    // High-pass filter
        
        // Framing & Windowing
        frames ← frame(y, win=25ms, hop=10ms)          // 1103 samples, 441 hop
        frames ← apply_hann_window(frames)
        
        // Spectral Analysis
        S ← STFT(frames, n_fft=2048)                   // Magnitude spectrum
        S_mel ← mel_filterbank(S, n_mels=40)           // Mel-scale energy
        S_log ← log(S_mel + 1e-10)                     // Log compression
        
        // Cepstral Coefficients
        mfcc ← DCT(S_log)[1:13]                        // Discard c₀, keep c₁..c₁₃
        delta ← compute_delta(mfcc, window=5)          // 1st derivative
        delta_delta ← compute_delta(delta, window=5)   // 2nd derivative
        
        // Final Feature Vector
        f ← concatenate([mfcc, delta, delta_delta])    // 39-dim per frame
        f ← normalize(f, μ_train, σ_train)             // Global speaker-independent norm
        
        // Sequence Length Standardization
        if len(f) < 120:
            f ← zero_pad(f, target=120)                // Pad shorter sequences
        elif len(f) > 120:
            f ← truncate(f, target=120)                // Truncate (rare: 0.3%)
        
        // Data Augmentation (Training Only, Stochastic)
        if training and random() < 0.5:
            f ← specaugment(f, T_mask=20, F_mask=5)    // Mask time/freq
        if training and random() < 0.5:
            f ← mixup(f, other_sample, α=0.2)          // Convex combination
    
    RETURN: Batch tensor X ∈ ℝ^(B×120×39)

================================================================================
MODEL ARCHITECTURE (Section 3.3):

    // ========== BRANCH 1: CNN-LSTM (Local Features) ==========
    
    FORWARD_CNN_LSTM(X):
        INPUT: X ∈ ℝ^(B×120×39)
        
        // 4-Layer CNN Stack (1D Convolutions)
        h₁ ← ReLU(BatchNorm(Conv1D(X, in=39, out=32, k=7, s=1)))
        h₂ ← ReLU(BatchNorm(Conv1D(h₁, in=32, out=64, k=3, s=1)))
        h₂ ← MaxPool(h₂, kernel=2, stride=2)              // 120 → 60 frames
        h₃ ← ReLU(BatchNorm(Conv1D(h₂, in=64, out=128, k=3, s=1)))
        h₄ ← ReLU(BatchNorm(Conv1D(h₃, in=128, out=256, k=3, s=1)))
        
        // 2-Layer Bidirectional LSTM
        h₅ ← BiLSTM_layer1(h₄, hidden=128, dropout=0.3)   // ∈ ℝ^(B×60×256)
        h₆ ← BiLSTM_layer2(h₅, hidden=128, dropout=0.3)   // ∈ ℝ^(B×60×256)
        
        // Attention-Based Temporal Pooling
        For t = 1 to 60:
            u_t ← tanh(W_a · h₆[t] + b_a)                 // ∈ ℝ^128
        α ← softmax([u₁^T·v, ..., u₆₀^T·v])              // Attention weights
        z_cnn_lstm ← Σ_t α_t · h₆[t]                      // ∈ ℝ^(B×256)
        
        RETURN: z_cnn_lstm ∈ ℝ^(B×256)
    
    // ========== BRANCH 2: TRANSFORMER (Global Context) ==========
    
    FORWARD_TRANSFORMER(x_waveform):
        INPUT: x_waveform ∈ ℝ^(B×44100) (raw audio for 1 sec)
        
        // Wav2Vec 2.0 Feature Extractor (Pre-trained on LibriSpeech)
        H_wav2vec ← wav2vec_cnn_encoder(x_waveform)       // ∈ ℝ^(B×120×768)
        
        // Positional Encoding (Sinusoidal)
        For t = 1 to 120, i = 0 to 383:
            PE[t, 2i]   ← sin(t / 10000^(2i/768))
            PE[t, 2i+1] ← cos(t / 10000^(2i/768))
        H₀ ← H_wav2vec + PE
        
        // 12-Layer Transformer Encoder
        For layer ℓ = 1 to 12:
            // Multi-Head Self-Attention (12 heads, d_k=64)
            Attn ← MultiHeadAttention(H_{ℓ-1}, heads=12)
            H̃_ℓ ← LayerNorm(H_{ℓ-1} + Attn)
            
            // Position-wise FFN (hidden=3072)
            FFN_out ← GELU(H̃_ℓ · W₁ + b₁) · W₂ + b₂
            H_ℓ ← LayerNorm(H̃_ℓ + FFN_out)
        
        // Global Average Pooling
        z_transformer ← mean(H₁₂, dim=time)               // ∈ ℝ^(B×768)
        
        RETURN: z_transformer ∈ ℝ^(B×768)
    
    // ========== FUSION & CLASSIFICATION HEAD ==========
    
    FORWARD_CLASSIFIER(z_cnn_lstm, z_transformer):
        // Feature Fusion via Concatenation
        z_fused ← concatenate([z_cnn_lstm, z_transformer])  // ∈ ℝ^(B×1024)
        
        // 2-Layer FC with Dropout (0.5)
        h₇ ← Dropout(ReLU(BatchNorm(FC(z_fused, 1024→512))), p=0.5)
        h₈ ← Dropout(ReLU(BatchNorm(FC(h₇, 512→256))), p=0.5)
        
        // Output Layer (Softmax)
        logits ← FC(h₈, 256→20)                           // ∈ ℝ^(B×20)
        probs ← Softmax(logits)                           // ∈ ℝ^(B×20)
        
        RETURN: probs
    
    FULL_FORWARD_PASS(x_waveform, x_mfcc):
        z_cnn_lstm ← FORWARD_CNN_LSTM(x_mfcc)
        z_transformer ← FORWARD_TRANSFORMER(x_waveform)
        probs ← FORWARD_CLASSIFIER(z_cnn_lstm, z_transformer)
        RETURN: probs

================================================================================
LOSS COMPUTATION (Section 3.3.3):

    LOSS_FUNCTION(probs, y_true):
        // Cross-Entropy with Label Smoothing (ε = 0.1)
        For each sample b, class i:
            if i == y_true[b]:
                ỹ[b, i] ← 1 - 0.1 + 0.1/20 = 0.905
            else:
                ỹ[b, i] ← 0.1/20 = 0.005
        
        loss ← -(1/B) Σ_b Σ_i ỹ[b,i] · log(probs[b,i])
        RETURN: loss

================================================================================
TRAINING PROCEDURE (Section 3.4):

    INITIALIZE:
        θ ← random_init(seed=42)                          // All parameters
        θ_wav2vec ← load_pretrained("facebook/wav2vec2-base")
        
        // Optimizer: Adam
        optimizer ← Adam(θ, β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=1e-4)
        
        // LR Schedule: Cosine Annealing with Warm Restarts
        η_max ← 1e-4, η_min ← 1e-6, T_max ← 10
        scheduler ← CosineAnnealingWarmRestarts(η_max, η_min, T_max)
        
        best_val_acc ← 0
        patience_counter ← 0
    
    // ========== STAGE 1: Frozen Wav2Vec (Epochs 1-10) ==========
    For epoch = 1 to 10:
        freeze(θ_wav2vec)                                 // 31.2M params frozen
        set_lr(optimizer, cnn_lstm=1e-4, transformer_enc=2e-4)
        
        For batch (X_mfcc, X_waveform, Y) in DataLoader(D_train, batch=32, shuffle=True):
            // Forward Pass
            probs ← FULL_FORWARD_PASS(X_waveform, X_mfcc)
            loss ← LOSS_FUNCTION(probs, Y)
            
            // Backward Pass
            loss.backward()
            clip_grad_norm(θ, max_norm=1.0)               // Prevent exploding gradients
            
            // Gradient Accumulation (every 4 steps → effective batch=128)
            if step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            scheduler.step()                              // Update LR
        
        // Validation
        val_acc ← EVALUATE(D_val)
        if val_acc > best_val_acc:
            best_val_acc ← val_acc
            save_checkpoint(θ, "best_model_stage1.pth")
            patience_counter ← 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break                                         // Early stopping
    
    // ========== STAGE 2: Full Fine-Tuning (Epochs 11-50) ==========
    For epoch = 11 to 50:
        unfreeze(θ_wav2vec)                               // All 91.7M params trainable
        set_lr(optimizer, cnn_lstm=1e-4, classifier=1e-4, wav2vec=5e-5)  // Discriminative LR
        
        For batch (X_mfcc, X_waveform, Y) in DataLoader(D_train, batch=32, shuffle=True):
            // Same as Stage 1, but with unfrozen Wav2Vec
            probs ← FULL_FORWARD_PASS(X_waveform, X_mfcc)
            loss ← LOSS_FUNCTION(probs, Y)
            loss.backward()
            clip_grad_norm(θ, max_norm=1.0)
            
            if step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            scheduler.step()
        
        // Validation & Early Stopping
        val_acc ← EVALUATE(D_val)
        if val_acc > best_val_acc:
            best_val_acc ← val_acc
            save_checkpoint(θ, "best_model.pth")
            patience_counter ← 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    // Load Best Model
    θ* ← load_checkpoint("best_model.pth")

================================================================================
INFERENCE PROCEDURE (Section 2.5):

    DECODE(x_test):
        // Preprocess
        x_mfcc ← PREPROCESS(x_test)                       // Extract 39-dim MFCCs
        x_waveform ← x_test                               // Raw waveform
        
        // Forward Pass (No Gradient)
        with torch.no_grad():
            probs ← FULL_FORWARD_PASS(x_waveform, x_mfcc)
        
        // Predict Class (Argmax)
        ŷ ← argmax(probs)                                 // ∈ {0, 1, ..., 19}
        
        RETURN: ŷ, probs

================================================================================
EVALUATION (Section 3.5):

    EVALUATE(D_test):
        y_true ← []
        y_pred ← []
        
        For (x, y) in D_test:
            ŷ, probs ← DECODE(x)
            y_true.append(y)
            y_pred.append(ŷ)
        
        // Primary Metrics
        accuracy ← sum(y_true == y_pred) / len(y_test)
        precision_per_class ← precision(y_true, y_pred, average=None)
        recall_per_class ← recall(y_true, y_pred, average=None)
        f1_per_class ← f1(y_true, y_pred, average=None)
        macro_f1 ← mean(f1_per_class)
        weighted_f1 ← f1(y_true, y_pred, average='weighted')
        
        // Confusion Matrix
        conf_matrix ← confusion_matrix(y_true, y_pred)    // 20×20
        
        // Calibration Metrics
        ece ← expected_calibration_error(y_true, probs, bins=10)
        mce ← maximum_calibration_error(y_true, probs, bins=10)
        brier ← brier_score(y_true, probs)
        
        // Statistical Significance (vs. baseline)
        mcnemar_p ← mcnemar_test(y_pred_baseline, y_pred)
        
        RETURN: {accuracy, macro_f1, weighted_f1, conf_matrix, ece, mce, brier, mcnemar_p}

================================================================================
RETURN:
    θ*:      Trained model parameters (91.7M params, 366 MB checkpoint)
    Ŷ_test:  Predicted labels for 2,500 test samples
    metrics: {
        accuracy: 97.82% (Arabic test set, reported in paper)
        macro_f1: ~97.5%
        conf_matrix: 20×20 matrix
        ece: ~0.015 (well-calibrated)
        inference_time: 5.3 ms per sample (NVIDIA V100)
    }
```

---

## 4. Architecture Figure (PlotNeuralNet)

### 4.1 High-Level Description

The PlotNeuralNet diagram illustrates the **parallel dual-branch hybrid architecture** with the following key elements:

1. **Input Stage**: Raw audio waveform (44.1 kHz) and preprocessed MFCC features (39-dim)
2. **Branch 1 (Bottom Path — CNN-LSTM)**:
   - 4× 1D Convolutional layers (32→64→128→256 filters) with BatchNorm + ReLU
   - MaxPool after 2nd conv (120→60 frames)
   - 2× Bidirectional LSTM layers (hidden=128 per direction)
   - Attention-based temporal pooling → 256-dim vector
3. **Branch 2 (Top Path — Transformer)**:
   - Wav2Vec 2.0 CNN feature extractor (7 layers, pre-trained on LibriSpeech)
   - Sinusoidal positional encoding
   - 12-layer Transformer encoder (MHSA + FFN)
   - Global average pooling → 768-dim vector
4. **Fusion Stage**: Concatenate branch outputs → 1024-dim fused representation
5. **Classification Head**: 2× FC layers (1024→512→256) with Dropout (0.5) + BatchNorm
6. **Output**: 20-class softmax probabilities
7. **Loss**: Cross-entropy with label smoothing (ε=0.1)
8. **Annotations**: Tensor dimensions, layer counts, hyperparameters (12 heads, hidden dims)

**Diagram Conventions**:
- **Blue blocks**: CNN-LSTM branch components
- **Green blocks**: Transformer branch components
- **Orange blocks**: Shared/fusion components
- **Red block**: Output layer
- **Arrows**: Data flow (thick for main paths, dashed for attention/auxiliary)
- **Labels**: Tensor shapes (B × T × D notation)

### 4.2 Minimal Reproducible LaTeX Code

**Prerequisites**:
1. Download PlotNeuralNet: `git clone https://github.com/HarisIqbal88/PlotNeuralNet.git`
2. Ensure `init.tex` (PlotNeuralNet macros) is in the same directory as this file
3. Compile with: `pdflatex hybrid_architecture.tex`

**File: `hybrid_architecture.tex`**

```latex
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{./}{init}  % PlotNeuralNet initialization
\usetikzlibrary{positioning, arrows.meta, calc}

\begin{document}
\begin{tikzpicture}

% ===================================================================
% DEFINE CUSTOM STYLES
% ===================================================================
\tikzset{
    connection/.style={-Stealth, thick},
    attention/.style={-Stealth, thick, dashed, red!60},
    label/.style={font=\footnotesize, align=center}
}

% ===================================================================
% INPUT STAGE
% ===================================================================
\pic[shift={(0,0,0)}] at (0,0,0) 
    {Box={name=audio, caption=Raw Audio, fill=gray!20, 
          xlabel={}, ylabel=44.1 kHz, zlabel=1 sec, 
          width=2, height=2, depth=2}};

\pic[shift={(3,0,0)}] at (audio-east) 
    {Box={name=mfcc, caption=MFCC Features, fill=gray!30,
          xlabel={}, ylabel=$T{=}120$, zlabel=$D{=}39$,
          width=3, height=3, depth=6}};

% ===================================================================
% BRANCH 2 (TOP): TRANSFORMER PATH
% ===================================================================
% Wav2Vec 2.0 Feature Extractor
\pic[shift={(0,4,0)}] at (audio-east) 
    {Box={name=wav2vec, caption=Wav2Vec 2.0 CNN, fill=green!20,
          xlabel={Pre-trained}, ylabel=7 layers, zlabel=31.2M params,
          width=3, height=3, depth=5}};

% Positional Encoding
\pic[shift={(3,0,0)}] at (wav2vec-east) 
    {Box={name=posenc, caption=Positional Enc., fill=green!15,
          xlabel={Sinusoidal}, ylabel=$T{=}120$, zlabel=$d{=}768$,
          width=2, height=2.5, depth=5}};

% Transformer Encoder (Grouped as single block)
\pic[shift={(3,0,0)}] at (posenc-east) 
    {Box={name=transformer, caption=Transformer Encoder, fill=green!25,
          xlabel={12 layers}, ylabel=MHSA + FFN, zlabel=58.3M params,
          width=5, height=4, depth=10}};

% Annotation: Transformer details
\node[below=0.3cm of transformer-south, label, text width=4.5cm] 
    {\tiny 12 heads, $d_k{=}64$\\
     Hidden=3072, GELU\\
     LayerNorm + Residual};

% Global Average Pooling
\pic[shift={(3,0,0)}] at (transformer-east) 
    {Box={name=transpool, caption=Avg Pool, fill=green!15,
          xlabel={Global}, ylabel=, zlabel=$B{\times}768$,
          width=2, height=2, depth=4}};

% ===================================================================
% BRANCH 1 (BOTTOM): CNN-LSTM PATH
% ===================================================================
% CNN Layers (4× Conv1D)
\pic[shift={(3,-4,0)}] at (mfcc-east) 
    {Box={name=cnn, caption=CNN Layers, fill=blue!20,
          xlabel={4× Conv1D}, ylabel=32→64→128→256, zlabel=0.4M params,
          width=4, height=3.5, depth=7}};

% Annotation: CNN details
\node[above=0.2cm of cnn-north, label, text width=3.5cm] 
    {\tiny Kernels: [7,3,3,3]\\
     BatchNorm + ReLU\\
     MaxPool (120→60)};

% BiLSTM Layers (2× Stacked)
\pic[shift={(3,0,0)}] at (cnn-east) 
    {Box={name=bilstm, caption=BiLSTM, fill=blue!25,
          xlabel={2 layers}, ylabel=Hidden=128×2, zlabel=0.8M params,
          width=3.5, height=3.5, depth=7}};

% Attention Pooling
\pic[shift={(3,0,0)}] at (bilstm-east) 
    {Box={name=attnpool, caption=Attention Pool, fill=blue!15,
          xlabel={Learned}, ylabel=$\alpha_t$ weights, zlabel=$B{\times}256$,
          width=2.5, height=2, depth=4}};

% Attention mechanism annotation
\node[below=0.2cm of attnpool-south, label, text width=2.5cm] 
    {\tiny $u_t{=}\tanh(W_a h_t)$\\
     $\alpha_t{\propto}\exp(u_t^T v)$};

% ===================================================================
% FUSION & CLASSIFICATION HEAD
% ===================================================================
% Concatenation Node
\pic[shift={(2.5,0,0)}] at ($(transpool-east)!0.5!(attnpool-east)$) 
    {Box={name=fusion, caption=Concatenate, fill=orange!25,
          xlabel={}, ylabel=, zlabel=$B{\times}1024$,
          width=2, height=2.5, depth=6}};

% FC Layer 1
\pic[shift={(2.5,0,0)}] at (fusion-east) 
    {Box={name=fc1, caption=FC + Dropout, fill=orange!20,
          xlabel={1024→512}, ylabel=ReLU + BN, zlabel=Dropout=0.5,
          width=2.5, height=2.5, depth=5}};

% FC Layer 2
\pic[shift={(2.5,0,0)}] at (fc1-east) 
    {Box={name=fc2, caption=FC + Dropout, fill=orange!20,
          xlabel={512→256}, ylabel=ReLU + BN, zlabel=Dropout=0.5,
          width=2.5, height=2.5, depth=4}};

% Output Layer (Softmax)
\pic[shift={(2.5,0,0)}] at (fc2-east) 
    {Box={name=output, caption=Softmax, fill=red!30,
          xlabel={256→20}, ylabel=, zlabel=$B{\times}20$,
          width=2, height=2, depth=3}};

% Loss annotation
\node[below=0.3cm of output-south, label] 
    {\tiny Cross-Entropy\\
     Label Smoothing ε=0.1};

% ===================================================================
% CONNECTIONS (Data Flow)
% ===================================================================
% Input to branches
\draw [connection] (audio-east) -- (wav2vec-west) node[midway, above, label] {44.1 kHz};
\draw [connection] (mfcc-east) -- (cnn-west) node[midway, above, label] {120×39};

% Transformer path
\draw [connection] (wav2vec-east) -- (posenc-west);
\draw [connection] (posenc-east) -- (transformer-west);
\draw [connection] (transformer-east) -- (transpool-west);

% CNN-LSTM path
\draw [connection] (cnn-east) -- (bilstm-west);
\draw [connection] (bilstm-east) -- (attnpool-west);

% Fusion
\draw [connection] (transpool-east) -- (fusion-north) node[near end, above, label] {768};
\draw [connection] (attnpool-east) -- (fusion-south) node[near end, below, label] {256};

% Classification head
\draw [connection] (fusion-east) -- (fc1-west);
\draw [connection] (fc1-east) -- (fc2-west);
\draw [connection] (fc2-east) -- (output-west);

% Attention mechanism (dashed)
\draw [attention] (bilstm-north) to[bend left=20] (attnpool-north);

% ===================================================================
% ANNOTATIONS & LEGEND
% ===================================================================
% Title
\node at ($(transformer)+(0,2)$) [font=\Large\bfseries] 
    {Hybrid CNN-LSTM-Transformer for Arabic Word Recognition};

% Legend
\node at ($(audio)+(0,-3.5)$) [anchor=west, label, text width=12cm] 
    {\textbf{Legend:} 
     \textcolor{green!60!black}{\textbf{■}} Transformer Branch (global context) \quad
     \textcolor{blue!60!black}{\textbf{■}} CNN-LSTM Branch (local features) \quad
     \textcolor{orange!60!black}{\textbf{■}} Fusion + Classifier \quad
     \textcolor{red!60!black}{\textbf{■}} Output
    };

% Model Stats
\node at ($(audio)+(0,-4.5)$) [anchor=west, label, text width=12cm] 
    {\textbf{Model:} 91.7M params (97.6\% Transformer) \quad
     \textbf{Inference:} 5.3 ms (V100) \quad
     \textbf{Accuracy:} 97.82\% (Arabic test)
    };

% Two-stage training note
\node at ($(wav2vec)+(0,1.2)$) [anchor=south, label, text width=4cm, fill=yellow!10, draw] 
    {\textbf{Two-Stage Training:}\\
     \textbf{Epochs 1-10:} Freeze Wav2Vec\\
     \textbf{Epochs 11-50:} Fine-tune all};

\end{tikzpicture}
\end{document}
```

---

## 5. Step-by-Step Figure Design Guide

### Iterative Refinement Process

Follow these steps to build the architecture diagram from scratch or modify the provided template:

#### **Step 1: Start with Basic Pipeline Flow**
- **Goal**: Establish the overall data flow (left-to-right, top-to-bottom)
- **Actions**:
  1. Place **Input** node (raw audio) at left
  2. Add **MFCC** node to the right of audio
  3. Split into **two parallel branches**: Transformer (top) and CNN-LSTM (bottom)
  4. End with **Fusion** and **Output** nodes on the right
- **LaTeX Code**:
  ```latex
  \pic[shift={(0,0,0)}] at (0,0,0) {Box={name=audio, ...}};
  \pic[shift={(3,0,0)}] at (audio-east) {Box={name=mfcc, ...}};
  \pic[shift={(0,4,0)}] at (audio-east) {Box={name=wav2vec, ...}};
  \pic[shift={(3,-4,0)}] at (mfcc-east) {Box={name=cnn, ...}};
  ```

#### **Step 2: Add Branch-Specific Components**
- **Transformer Branch (Top)**:
  1. Wav2Vec 2.0 feature extractor → Positional encoding → Transformer encoder → Avg pool
  2. Use `shift={(x,y,0)}` to position relative to previous node's `-east` anchor
- **CNN-LSTM Branch (Bottom)**:
  1. CNN layers → BiLSTM → Attention pool
  2. Ensure vertical alignment with Transformer branch outputs (adjust y-coordinates)
- **LaTeX Code**:
  ```latex
  % Transformer path
  \pic[shift={(3,0,0)}] at (wav2vec-east) {Box={name=posenc, ...}};
  \pic[shift={(3,0,0)}] at (posenc-east) {Box={name=transformer, ...}};
  
  % CNN-LSTM path
  \pic[shift={(3,0,0)}] at (cnn-east) {Box={name=bilstm, ...}};
  ```

#### **Step 3: Annotate Tensor Shapes**
- **Goal**: Show dimensionality at each stage for clarity
- **Actions**:
  1. Use `xlabel`, `ylabel`, `zlabel` attributes in `Box` definitions
  2. Convention: `xlabel` for bottom label (e.g., layer config), `ylabel` for side (e.g., temporal dim), `zlabel` for top (e.g., feature dim)
  3. Use LaTeX math mode: `$T{=}120$`, `$B{\times}768$`
- **Example**:
  ```latex
  {Box={name=mfcc, xlabel={}, ylabel=$T{=}120$, zlabel=$D{=}39$, ...}}
  {Box={name=fusion, xlabel={}, ylabel=, zlabel=$B{\times}1024$, ...}}
  ```

#### **Step 4: Parameterize Architecture Details**
- **Goal**: Communicate hyperparameters (layer counts, hidden dims, etc.)
- **Actions**:
  1. Add `\node` annotations below/above boxes
  2. Use `\tiny` or `\footnotesize` fonts for compactness
  3. Include: # layers (L=12), # heads (h=12), hidden dimensions (d_model=768), dropout rates
- **Example**:
  ```latex
  \node[below=0.3cm of transformer-south, label, text width=4.5cm] 
      {\tiny 12 heads, $d_k{=}64$\\
       Hidden=3072, GELU\\
       LayerNorm + Residual};
  ```

#### **Step 5: Add Loss and Training Labels**
- **Goal**: Clarify the objective function and training strategy
- **Actions**:
  1. Place loss annotation near **Output** node
  2. Add two-stage training note near **Wav2Vec** block (frozen → fine-tuned)
- **Example**:
  ```latex
  \node[below=0.3cm of output-south, label] 
      {\tiny Cross-Entropy\\
       Label Smoothing ε=0.1};
  
  \node[anchor=south, label, fill=yellow!10, draw] at ($(wav2vec)+(0,1.2)$)
      {\textbf{Two-Stage Training:}\\
       Epochs 1-10: Freeze Wav2Vec\\
       Epochs 11-50: Fine-tune all};
  ```

#### **Step 6: Draw Connections (Data Flow)**
- **Goal**: Show how information propagates through the network
- **Actions**:
  1. Use `\draw [connection]` for standard data flow (solid arrows)
  2. Use `\draw [attention]` for attention mechanisms (dashed arrows)
  3. Add `node[midway, above/below, label]` for connection labels (e.g., tensor shapes)
- **LaTeX Code**:
  ```latex
  \draw [connection] (audio-east) -- (wav2vec-west) node[midway, above] {44.1 kHz};
  \draw [connection] (transpool-east) -- (fusion-north) node[near end, above] {768};
  \draw [attention] (bilstm-north) to[bend left=20] (attnpool-north);
  ```

#### **Step 7: Color Coding and Visual Hierarchy**
- **Goal**: Use color to distinguish components and guide attention
- **Actions**:
  1. **Green**: Transformer branch (global context)
  2. **Blue**: CNN-LSTM branch (local features)
  3. **Orange**: Fusion and classifier (shared components)
  4. **Red**: Output layer (final predictions)
  5. **Gray**: Input preprocessing
  6. Adjust `fill` attribute in `Box` definitions
- **Example**:
  ```latex
  {Box={name=transformer, fill=green!25, ...}}
  {Box={name=cnn, fill=blue!20, ...}}
  {Box={name=fusion, fill=orange!25, ...}}
  {Box={name=output, fill=red!30, ...}}
  ```

#### **Step 8: Add Legend and Model Stats**
- **Goal**: Provide quick reference for colors and overall model properties
- **Actions**:
  1. Create legend node below diagram with colored squares (■) and descriptions
  2. Add model statistics: total params, inference time, accuracy
- **LaTeX Code**:
  ```latex
  \node[anchor=west, label] at ($(audio)+(0,-3.5)$)
      {\textbf{Legend:} 
       \textcolor{green!60!black}{\textbf{■}} Transformer \quad
       \textcolor{blue!60!black}{\textbf{■}} CNN-LSTM \quad ...};
  
  \node[anchor=west, label] at ($(audio)+(0,-4.5)$)
      {\textbf{Model:} 91.7M params \quad
       \textbf{Inference:} 5.3 ms \quad ...};
  ```

#### **Step 9: Refine Aesthetics and Alignment**
- **Goal**: Ensure readability and professional appearance
- **Actions**:
  1. **Align nodes**: Use consistent `shift` values (e.g., x=3.0 between nodes)
  2. **Avoid overlaps**: Adjust `depth`, `width`, `height` of `Box` to prevent collisions
  3. **Font sizes**: Use `\footnotesize` for labels, `\tiny` for detailed annotations
  4. **Spacing**: Add `below=0.Xcm` or `above=0.Xcm` in `\node` positions
  5. **Test compile**: Run `pdflatex` and inspect; iterate as needed
- **Tips**:
  - If nodes overlap, increase `shift` x-coordinates
  - If text is cramped, reduce `text width` in `\node` or increase box dimensions
  - Use `\tikzset` to define global styles (e.g., `connection/.style={-Stealth, thick}`)

#### **Step 10: Validate and Export**
- **Goal**: Ensure diagram accurately reflects methodology
- **Actions**:
  1. Cross-check tensor dimensions against algorithm (Section 3)
  2. Verify parameter counts match Table in Section 3.3.4
  3. Confirm two-stage training annotation matches Section 3.4.1
  4. Ensure all 12 Transformer layers, 4 CNN layers, 2 LSTM layers are mentioned
  5. Export as PDF: `pdflatex hybrid_architecture.tex`
  6. Convert to PNG if needed: `pdftocairo -png -r 300 hybrid_architecture.pdf`
- **Validation Checklist** (see Section 6 below)

---

### Quick-Start Modifications

**To adjust for different architectures**:

1. **Change layer counts**: Modify `xlabel` (e.g., `12 layers` → `6 layers`)
2. **Update dimensions**: Edit `ylabel`, `zlabel` (e.g., `$d{=}768$` → `$d{=}512$`)
3. **Add/remove components**: Insert/delete `\pic` blocks and corresponding `\draw` connections
4. **Swap fusion method**: Replace "Concatenate" box with "Gated Fusion" or "Attention Fusion"
5. **Streaming/online**: Add "Chunking" and "Cache" annotations near Transformer encoder

---

## 6. Validation Checklist

### 6.1 Consistency Between Text, Algorithm, and Figure

- [ ] **Input dimensions match**:
  - Text (Section 3.2): MFCCs are 39-dim, 120 frames
  - Algorithm: `X ∈ ℝ^(B×120×39)`
  - Figure: MFCC box labeled `ylabel=$T{=}120$, zlabel=$D{=}39$`

- [ ] **Branch 1 (CNN-LSTM) flow**:
  - Text (Section 3.3.1): 4 Conv layers (32→64→128→256), MaxPool after 2nd, 2× BiLSTM (hidden=128), attention pool → 256-dim
  - Algorithm: `h₄ ∈ ℝ^(B×60×256)`, `z_cnn_lstm ∈ ℝ^(B×256)`
  - Figure: CNN box shows "32→64→128→256", BiLSTM "Hidden=128×2", attnpool "$B{\times}256$"

- [ ] **Branch 2 (Transformer) flow**:
  - Text (Section 3.3.2): Wav2Vec 7-layer CNN, 12-layer Transformer (12 heads, d_k=64, FFN hidden=3072), global avg pool → 768-dim
  - Algorithm: `H₁₂ ∈ ℝ^(B×120×768)`, `z_transformer ∈ ℝ^(B×768)`
  - Figure: Transformer box "12 layers, 12 heads", transpool "$B{\times}768$"

- [ ] **Fusion mechanism**:
  - Text (Section 3.3.3): Concatenation, 256 + 768 = 1024
  - Algorithm: `z_fused = [z_cnn_lstm ; z_transformer] ∈ ℝ^(B×1024)`
  - Figure: Fusion box "$B{\times}1024$", arrows from 768 and 256 branches

- [ ] **Classification head**:
  - Text (Section 3.3.3): 1024→512→256→20, Dropout=0.5
  - Algorithm: `h₇ ∈ ℝ^(B×512)`, `h₈ ∈ ℝ^(B×256)`, `y ∈ ℝ^(B×20)`
  - Figure: FC boxes "1024→512", "512→256", output "256→20", dropout annotated

- [ ] **Loss function**:
  - Text (Section 3.3.3): Cross-entropy with label smoothing ε=0.1
  - Algorithm: `LOSS_FUNCTION` with ỹ[b,i] = 0.905 (true) or 0.005 (false)
  - Figure: Loss annotation below output "Label Smoothing ε=0.1"

- [ ] **Training strategy**:
  - Text (Section 3.4.1): Two-stage (epochs 1–10 frozen, 11–50 fine-tuned)
  - Algorithm: `Stage 1` and `Stage 2` blocks with freeze/unfreeze logic
  - Figure: Yellow note box near Wav2Vec "Epochs 1-10: Freeze, 11-50: Fine-tune"

### 6.2 Dimension Flow Verification

- [ ] **Trace forward pass manually**:
  1. Start: Raw audio (44,100 samples) + MFCCs (120×39)
  2. Branch 1: 120×39 → CNN → 60×256 → BiLSTM → 60×256 → Attn → 256
  3. Branch 2: 44,100 → Wav2Vec → 120×768 → +PE → 120×768 → 12×Transformer → 120×768 → AvgPool → 768
  4. Fusion: [256 ; 768] → 1024
  5. Classifier: 1024 → 512 → 256 → 20
  6. **No dimension mismatches?** ✓

- [ ] **Parameter counts consistent**:
  - Text (Table): CNN-LSTM 1.2M, Transformer 89.5M, Classifier 1.0M, Total 91.7M
  - Figure: Annotate params in boxes (e.g., "31.2M params" for Wav2Vec)
  - Algorithm: Matches text

### 6.3 Loss and Decoding Path

- [ ] **Loss head present**: Cross-entropy node/annotation near output
- [ ] **No CTC/AED/RNN-T**: Confirm figure does NOT show sequence-to-sequence decoding (this is classification)
- [ ] **Softmax output**: Output layer explicitly labeled "Softmax", produces 20 class probabilities
- [ ] **Inference path clear**: Arrows flow from input → branches → fusion → classifier → output

### 6.4 Visual Quality

- [ ] **No overlapping nodes**: All boxes are spatially separated
- [ ] **Readable fonts**: Labels are ≥\tiny size, not too cramped
- [ ] **Colors distinguishable**: Green (Transformer), Blue (CNN-LSTM), Orange (fusion), Red (output) are visually distinct
- [ ] **Arrows well-routed**: Connections don't cross unnecessarily; use `bend left/right` if needed
- [ ] **Legend present**: Color legend + model stats at bottom
- [ ] **Title clear**: "Hybrid CNN-LSTM-Transformer for Arabic Word Recognition" at top

### 6.5 Compilation and Export

- [ ] **Compiles without errors**: `pdflatex hybrid_architecture.tex` succeeds
- [ ] **PlotNeuralNet macros loaded**: `init.tex` is in the same directory
- [ ] **PDF output clean**: No artifacts, text is crisp
- [ ] **High-resolution export** (if needed): Use `-r 300` for PNG conversion

---

## 7. Compile Instructions

### 7.1 Requirements

1. **LaTeX distribution**: TeX Live 2020+ or MiKTeX
2. **PlotNeuralNet**: Clone from [GitHub](https://github.com/HarisIqbal88/PlotNeuralNet)
   ```bash
   git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
   cd PlotNeuralNet/pycore
   ```
3. **File structure**:
   ```
   project/
   ├── hybrid_architecture.tex  (diagram LaTeX file)
   ├── init.tex                 (PlotNeuralNet macros, from repo)
   └── layers/                  (PlotNeuralNet layer definitions, from repo)
   ```

### 7.2 Compilation

**Basic compilation**:
```bash
pdflatex hybrid_architecture.tex
```

**If errors occur**:
- Ensure `init.tex` is present: `ls init.tex`
- Check TikZ packages installed: `tlmgr install pgf tikz standalone`
- View log: `less hybrid_architecture.log`

**Generate high-res PNG** (optional):
```bash
pdflatex hybrid_architecture.tex
pdftocairo -png -r 300 hybrid_architecture.pdf hybrid_architecture.png
```
- `-r 300`: 300 DPI resolution (publication quality)
- Output: `hybrid_architecture-1.png`

### 7.3 Troubleshooting

| Error | Solution |
|-------|----------|
| `File 'init.tex' not found` | Copy `init.tex` from PlotNeuralNet repo to working directory |
| `Undefined control sequence \pic` | Install TikZ: `tlmgr install pgf tikz` |
| `Package standalone Error` | Install standalone: `tlmgr install standalone` |
| Nodes overlap | Increase `shift={(x,0,0)}` values (e.g., 3.0 → 3.5) |
| Text too small | Change `\tiny` to `\footnotesize` in node annotations |

---

## 8. Summary and Recommendations

### 8.1 Methodology Strengths

1. **Comprehensive documentation**: Every hyperparameter, dataset split, and training detail is specified
2. **Reproducibility**: Fixed seeds, public code, pre-trained models available
3. **Rigorous evaluation**: Multiple metrics (accuracy, F1, ECE, MCE, Brier), statistical tests, robustness checks
4. **Novel architecture**: Parallel dual-branch design (not sequential) with attention-based pooling
5. **Transfer learning**: Effective two-stage fine-tuning of pre-trained Wav2Vec 2.0

### 8.2 Recommendations for Future Work

1. **Extend to continuous ASR**: Current work is isolated words; generalize to phrase/sentence recognition with CTC/AED losses
2. **Dialectal robustness**: Train on multi-dialectal Arabic (current: Tunisian only)
3. **Gender diversity**: Include female speakers (current: 100% male)
4. **Real-time streaming**: Adapt Transformer for online decoding (chunked attention)
5. **Model compression**: Investigate knowledge distillation or pruning to reduce 91.7M params
6. **Multilingual**: Evaluate cross-lingual transfer (Arabic ↔ other languages)

### 8.3 Diagram Usage Notes

- **For papers**: Include as Figure 1 (architecture overview) in methodology section
- **For presentations**: Export as high-res PNG; add slide title "Hybrid Model Architecture"
- **For posters**: Scale up to A0 size; use larger fonts (change `\tiny` → `\small`)
- **For documentation**: Embed PDF in technical reports; link to interactive version

---

## Appendix: Alternative Diagram Variants

### A.1 Simplified Version (High-Level Overview)

For presentations with limited space, create a simplified diagram:
- Merge CNN layers into single "CNN Stack" block
- Merge BiLSTM layers into single "BiLSTM" block
- Remove detailed annotations (keep only main paths)
- Enlarge fonts to `\normalsize`

### A.2 Detailed Version (With Equations)

For technical documentation, add:
- LSTM gate equations inside BiLSTM block
- Self-attention formula inside Transformer block
- Loss function equation below output
- Gradient flow arrows (in dashed style)

### A.3 Comparison Diagram

Create side-by-side comparison with baseline models:
- **Left**: Sequential CNN→LSTM→Transformer (baseline)
- **Right**: Parallel dual-branch (proposed)
- Highlight differences with red/green annotations

---

**End of Extraction**

This document provides a complete, self-contained extraction of the methodology with:
✓ Comprehensive method summary (all details from methodology.tex)
✓ End-to-end algorithm (training + inference pseudocode)
✓ Production-ready PlotNeuralNet LaTeX code
✓ Step-by-step figure design guide (10 iterative steps)
✓ Validation checklist (dimension flow, consistency checks)
✓ Compile instructions with troubleshooting

All information is cited by section/line numbers from the original `methodology.tex`.
