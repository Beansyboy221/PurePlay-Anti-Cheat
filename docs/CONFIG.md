# Updated Configuration Guide

This document provides an overview of the configuration options, input data properties, and training parameters used by the system. The system supports two model types: **unsupervised** and **supervised**. It is organized around four primary modes of operation. The configuration is defined in a JSON file (`config.json`).

## IMPORTANT

I have removed all metadata transfer between modes. It is up to you to make sure you match your polling rate, sequence length, and whitelists between modes, data, and models!

---

## 1. Overview

The system operates in one of four modes:

- **Data Collection (`"collect"`):**  
  Captures input data from keyboard, mouse, and gamepad devices at a specified polling rate and saves the data as CSV files.

- **Model Training (`"train"`):**  
  Loads one or more CSV files containing input sequences and trains a model. Supported model types include:
  - **Unsupervised:** Reconstructs input sequences given lots non-cheating data.
  - **Supervised:** Classifies input sequences given lots of negative and positive data.

- **Static Analysis (`"test"`):**  
  Uses a pre-trained model to analyze a selected CSV file, computes relevant metrics (such as reconstruction error, classification confidence, or prediction loss), and generates a report graph.

- **Live Analysis (`"deploy"`):**  
  Continuously polls the input devices in real time, accumulates sequences, processes them using a pre-trained model, and prints a stream of metrics for external analysis.

The mode is set by the `"mode"` key in `config.json`.

---

## 2. Configuration File: `config.json`

- **mode**  
  *Type:* String  
  *Description:* Selects the operational mode.  
  *Possible Values:*  
  - `"collect"` — Data Collection Mode  
  - `"train"` — Model Training Mode  
  - `"test"` — Static Analysis Mode  
  - `"deploy"` — Live Analysis Mode  

- **kill_key**  
  *Type:* String  
  *Recommended:* `"\\"`  
  *Description:* The keyboard key that, when pressed, stops the program. For possible values, see `keyboard_whitelist`.

- **capture_bind**  
  *Type:* String  
  *Recommended:* `"right"`  
  *Description:* The button that, when pressed, starts data collection. Can be any button from the keyboard or mouse.

- **model_type**  
  *Type:* String  
  *Recommended:* `"unsupervised"`  
  *Description:* Pick supervised if you have a specific cheat you want to catch or lots of positive cheating data and legitimate player data. Otherwise, pick unsupervised and only train on normal player data.  
  *Possible Values:*  
  - `"unsupervised"`  
  - `"supervised"`  

- **keyboard_whitelist**  
  *Type:* List of Strings  
  *Recommended:* `["w", "a", "s", "d", "space", "ctrl"]`  
  *Description:* List of keyboard keys to capture during collection and live analysis.  
  *Possible Values:*  
    `a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, +, -, *, /, ., ,, <, >, ?, !, @, #, $, %, ^, &, *, (, ), _, =, {, }, [, ], |, \\, :, ;, , , ~, enter, esc, backspace, tab, space, caps lock, num lock, scroll lock, home, end, page up, page down, insert, delete, left, right, up, down, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, print screen, pause, break, windows, menu, right alt, ctrl, left shift, right shift, left windows, left alt, right windows, alt gr, windows, alt, shift, right ctrl, left ctrl`

- **mouse_whitelist**  
  *Type:* List of Strings  
  *Recommended:* `["angle", "magnitude"]`  
  *Description:* List of mouse features to capture.  
  *Possible Values:*  
    `left, right, middle, x, x2, angle, magnitude`  

- **gamepad_whitelist**  
  *Type:* List of Strings  
  *Recommended:* `["LT", "RT", "LX", "LY", "RX", "RY"]`  
  *Description:* List of gamepad buttons/features to capture.  
  *Possible Values:*  
    `DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, START, BACK, LEFT_THUMB, RIGHT_THUMB, LEFT_SHOULDER, RIGHT_SHOULDER, A, B, X, Y, LT, RT, LX, LY, RX, RY`

- **polling_rate**  
  *Type:* Integer (Hz)  
  *Recommended:* `90`  
  *Description:* The frequency at which keyboard, mouse, and gamepad inputs are polled.

- **sequence_length**  
  *Type:* Integer  
  *Recommended:* `30`  
  *Description:* The number of polls per input pattern you want to recognize.

- **batch_size**  
  *Type:* Integer  
  *Recommended:* `64`  
  *Description:* Number of sequences per training batch.

---

## 3. Understanding Feature Selection

### What is Feature Selection?

Feature selection is a critical process in model training. It involves choosing the most relevant input features from your available data (keyboard, mouse, or gamepad) to help the model learn meaningful patterns. Effective feature selection can:

- **Reduce Dimensionality:** By limiting the number of features, the model trains faster and requires fewer computational resources.
- **Minimize Noise:** Selecting only the most pertinent features reduces the risk of the model learning from irrelevant or redundant data.
- **Improve Accuracy:** A carefully chosen set of features helps the model focus on the key signals, leading to better generalization and improved performance on unseen data.

### Best Practices for Feature Selection

- **Leverage Domain Knowledge:**  
  Use insights about your application to select features that are most likely to contribute valuable information. For instance, specific keys or mouse movements might be more indicative of user behavior in your use case. They may also be more prone to cheating inputs such as auto-strafe or anti-recoil macros.
- **Iterative Refinement:**  
  Start with a broad set of features and evaluate the model's performance. Gradually remove or adjust features based on validation results, focusing on those that have the most impact.
- **Monitor for Overfitting:**  
  Including too many features can lead to overfitting. A lean, well-chosen feature set helps the model generalize better.
- **Reduce Redundancy:**  
  Analyze your features for high correlations. Eliminating redundant or highly correlated features can simplify the training process and improve model stability.

### Impact on Model Accuracy and Performance

- **High-Quality Features:**  
  When the model is trained on carefully selected features, it can more easily capture the essential patterns in the data, leading to improved accuracy and faster convergence.
- **Irrelevant or Excessive Features:**  
  Including features that add little value or introduce excessive noise may hinder the training process, slow down convergence, and negatively affect overall performance.

---

## 4. Analyzing Graph Outputs

The training and evaluation process generates three distinct types of graphs. Each graph plots metrics over the sequence indices (time) and provides insights into different aspects of model performance.

### 4.1 Reconstruction Error Graph

- **What It Shows:**  
  This graph plots the reconstruction error when using the `"unsupervised"` model.
- **How to Interpret:**  
  - **Low Reconstruction Error:** Suggests that the autoencoder recognizes the player's behavior as normal relative to your training data.
  - **High Reconstruction Error:** May indicate the randomness of normal player behavior. Frequent spikes may indicate poor training data or a cheating player.
- **Use Case:**  
  This method is generally easier and ideal as it requires only negative training data.

### 4.2 Targeted Class Confidence Graph

- **What It Shows:**  
  This graph presents the `"supervised"` model’s confidence (or probability) that the player is cheating.
- **How to Interpret:**  
  The higher the line, the higher the confidence that the player is cheating. You may want to determine a threshold for automatic flagging.
- **Use Case:**  
  As a supervised approach, this is useful for when you can isolate and identify specific cheating input patterns.
  
---

## 5. Best Practices for Intelligent Training

- **Data Quality:**  
  Verify that your data is representative of normal player behavior. This can be tricky considering player behavior is naturally random.
  
- **Iterative Refinement:**  
  Start with moderate settings for parameters such as `sequence_length` and `batch_size`. Use early training results (observed via the graphs) to iteratively refine the configuration.
  
- **Balanced Feature Selection:**  
  Experiment with different feature whitelists to determine which inputs contribute most effectively to the model’s performance. Use the graph insights to decide if certain features may be leading to increased loss or reconstruction error.
  
- **Resource Consideration:**  
  Understand the trade-off between model accuracy and computational cost. While larger window sizes and more files may improve accuracy, they also demand more resources when training and analyzing.
  
- **Continuous Monitoring:**  
  Use the three types of graphs (prediction loss, reconstruction error, and targeted class confidence) in live analysis mode to continually assess and adjust your training strategy.

---

## 6. Conclusion

This documentation provides a comprehensive guide to the configuration settings and the rationale behind them. By carefully curating input features and monitoring performance through the detailed graph outputs, you can optimize the system to meet your specific needs.

---
