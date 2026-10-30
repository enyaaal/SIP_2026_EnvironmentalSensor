# Classical vs. Semantic Communications for IoT Occupancy Detection

This repository contains the Jupyter Notebook implementation for a comparison of classical and semantic communication systems on the UCI Occupancy Detection dataset.

The project studies whether transmitting a compressed, task-oriented representation can reduce communication overhead while preserving useful occupancy-prediction performance over a noisy wireless channel.

## Project Overview

Two communication pipelines are implemented:

- **Classical communication:** quantizes four environmental sensor measurements to 16 bits each, for 64 bits per sample. The receiver reconstructs the measurements and uses a logistic regression classifier to predict occupancy.
- **Semantic communication:** uses a neural network encoder to map the four sensor measurements to a four-dimensional latent representation. Each latent value is quantized to 8 bits, for 32 bits per sample. A neural network decoder predicts occupancy from the received representation.

Both systems are evaluated over an additive white Gaussian noise (AWGN) channel. The notebook compares bit error rate (BER) and occupancy-prediction accuracy across signal-to-noise ratios (SNRs), modulation schemes, and channel-coding methods.

## Repository Contents

```text
.
├── README.md
├── sipeval.ipynb
├── datatraining.txt
└── datatest.txt
```

The two dataset files must be placed in the same directory as the notebook. If your repository uses a different folder structure, update the file paths in the notebook accordingly.

## Dataset

The notebook expects the UCI Occupancy Detection dataset in two text files:

- `datatraining.txt` — training data
- `datatest.txt` — test data

The features used are:

- `Temperature`
- `Humidity`
- `Light`
- `CO2`

The target column is `Occupancy`, where the dataset uses labels for occupied and unoccupied conditions.

The notebook reads the files with `pandas.read_csv()`. Make sure the files are available locally before running the notebook. The dataset is not included in this repository unless you add the files yourself.

## Methods

### Classical Communication

1. Fit a logistic regression occupancy classifier using the training measurements.
2. Quantize each of the four sensor features to 16 bits (64 payload bits per sample).
3. Apply channel coding:
   - Repetition coding (rate 1/3)
   - Hamming(7,4) coding
   - Convolutional coding with Viterbi decoding
4. Modulate the coded bits using BPSK, QPSK, or 16-QAM.
5. Transmit the modulated signal through an AWGN channel.
6. Demodulate and decode the received signal.
7. Reconstruct the sensor measurements and evaluate occupancy-prediction accuracy.

### Semantic Communication

1. Standardize the four input features using a `StandardScaler` fitted on the training data.
2. Train a neural network with an encoder and an occupancy-prediction decoder.
3. The encoder maps the four input features to a four-dimensional latent representation.
4. Quantize each latent value to 8 bits (32 payload bits per sample).
5. Transmit the latent bits through the AWGN channel using the communication functions in the notebook.
6. Decode and dequantize the received bits.
7. Use the neural network decoder to predict occupancy.

The semantic model is implemented in PyTorch. The notebook's main configuration uses 200 training epochs and the Adam optimizer with a learning rate of 0.01.

## Experimental Settings

| Component | Configurations |
|---|---|
| Channel | AWGN |
| SNR range | -10 dB to 20 dB, in 2 dB increments |
| Modulation | BPSK, QPSK, 16-QAM |
| Channel coding | Repetition, Hamming(7,4), convolutional coding with Viterbi decoding |
| Classical payload | 64 bits per sample |
| Semantic payload | 32 bits per sample |
| Semantic dimension | 4 |
| Bits per latent value | 8 |
| Semantic model training | 200 epochs; Adam; learning rate 0.01 |

The notebook also includes ablation experiments that vary the channel-coding method, modulation scheme, and semantic representation dimension.

## Installation

Python 3.9 or later is recommended.

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib scikit-learn torch jupyter
```

If you use JupyterLab, you can install it with:

```bash
python -m pip install jupyterlab
```

## Running the Notebook

1. Clone or download this repository.
2. Place `datatraining.txt` and `datatest.txt` in the repository directory.
3. Install the dependencies listed above.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

   Or, to use JupyterLab:

   ```bash
   jupyter lab
   ```

5. Open `sipeval.ipynb`.
6. Run the notebook cells in order.

The notebook uses CPU execution for the semantic model by default. Experiments that sweep across multiple SNR values, coding schemes, and modulation schemes may take some time to finish.

## Outputs

Depending on which cells are run, the notebook generates plots and summaries including:

- BER versus SNR
- Occupancy-prediction accuracy versus SNR
- Modulation comparison plots
- Channel-coding ablation plots
- Semantic-dimension ablation results
- Payload-bit and compression comparisons

Plots are saved as PNG files in the notebook's working directory by the corresponding cells.

## Reproducibility Notes

- The notebook sets NumPy and PyTorch random seeds to `0` in the semantic-model section. AWGN experiments use random noise, so results may still depend on execution order and random-number state.
- Run the notebook from top to bottom to ensure that data, models, functions, and result variables are defined.
- The notebook contains exploratory and repeated implementation cells. Some cells rerun experiments or redefine functions. For the most consistent results, execute the cells in their existing order and use the final experiment and plotting sections for the reported comparisons.
- The reported payload sizes (64 bits and 32 bits per sample) describe uncoded data payloads. Channel coding adds additional transmitted bits.
- The notebook evaluates simulated AWGN-channel performance. It does not, by itself, measure real-world wireless performance, energy consumption, or end-to-end latency.

## Research Context

This notebook supports the SIP 2026 project **“Classical vs. Semantic Communications for IoT Occupancy Detection.”** The central question is whether a communication system can transmit less information while retaining sufficient information for a downstream occupancy-detection task.

The project results should be interpreted as a trade-off rather than as evidence that semantic communication is universally better: the semantic system reduces the payload size, while the classical system can achieve higher prediction accuracy under favorable channel conditions.
