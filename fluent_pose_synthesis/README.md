# Fluent Sign Language Pose Synthesis
### Neural Post-Editing for Natural, Temporally Coherent Sign Language Motion

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Sign%20Language%20AI-blue">
  <img src="https://img.shields.io/badge/Task-Pose%20Synthesis-purple">
  <img src="https://img.shields.io/badge/Model-Deep%20Learning-orange">
  <img src="https://img.shields.io/badge/Data-DGS%20Corpus-green">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
</p>

## Overview

**Fluent Sign Language Pose Synthesis** is a deep-learning system for improving the fluency of automatically generated sign language pose sequences.

Sign language generation systems can construct sentences by concatenating isolated signs retrieved from dictionaries, videos, or symbolic representations such as HamNoSys. Although the individual signs may be correct, directly joining them often produces motion that is visually discontinuous and unnatural.

This project addresses that problem at the **pose-sequence level**.

The system learns to transform non-fluent or dictionary-composed sign sequences into smoother, temporally coherent signing while preserving the linguistic content of the original sequence.

The core objective is:

```text
Unfluent Pose Sequence
        +
Dictionary / Corrected Sign Poses
        │
        ▼
Temporal Alignment
        │
        ▼
Neural Pose Processing
        │
        ▼
Motion Refinement
        │
        ▼
Fluent Sign Language Pose Sequence
```

The project focuses on correcting aspects of generated signing such as:

- Temporal discontinuities
- Abrupt transitions between signs
- Incorrect sign duration
- Motion timing
- Prosodic inconsistencies
- Intonation-related pose behavior
- Dictionary-to-sentence transition artifacts

---

# The Problem

Modern sign language generation systems can produce signing from spoken-language input using pipelines such as:

```text
Spoken Language
      │
      ▼
Sign Language Translation
      │
      ▼
Gloss Sequence
      │
      ▼
Dictionary Sign Retrieval
      │
      ▼
Pose Sequences
      │
      ▼
Generated Signing
```

For example:

```text
"We were expecting something simple, like a youth hostel."
```

may be translated into a gloss sequence such as:

```text
DIFFERENT1 IMAGINATION1A LIKE3B* EASY1 YOUNG1* HOME1A
```

Each gloss can then be represented using an existing sign video, pose sequence, or HamNoSys representation.

The problem appears when those isolated signs are concatenated.

```text
Sign A       Sign B       Sign C       Sign D

  │            │            │            │
  └────────────┴────────────┴────────────┘
                       │
                       ▼
              Direct Concatenation
                       │
                       ▼
             Non-Fluent Motion
```

A dictionary sign is typically produced in isolation.

Natural sentence-level signing, however, contains continuous motion, contextual timing, transitions, prosody, and coarticulation.

As a result:

> Correct individual signs do not automatically produce a fluent sign language sentence.

This project is designed specifically to address that gap.

---

# Core Idea

Instead of generating an entire sign language sentence again from scratch, the system treats fluency improvement as a **pose-sequence post-editing problem**.

The proposed workflow uses:

1. An original sentence-level pose sequence.
2. Dictionary-form replacements for selected signs.
3. Temporal processing to align the replacement with the original sequence.
4. A neural model to reconstruct a fluent version of the modified sequence.

Conceptually:

```text
          Original Fluent Sentence
                    │
                    │
                    ▼
           Select Sign Segment
                    │
                    ▼
         Replace With Dictionary
               Sign Pose
                    │
                    ▼
       Artificially Non-Fluent
             Pose Sequence
                    │
                    ▼
             Neural Model
                    │
                    ▼
       Reconstructed Fluent Pose
                    │
                    ▼
          Original Sentence
             as Target
```

This formulation creates training pairs automatically:

```text
Non-Fluent / Modified Sequence → Fluent Original Sequence
```

and allows the model to learn how isolated signs should be integrated into continuous signing.

---

# System Pipeline

The full pipeline can be represented as:

```text
                        DGS Corpus
                            │
                            ▼
                 Sentence-Level Signing
                            │
                            ▼
                     Pose Extraction
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        Original Pose Sequence     Gloss Annotation
                                        │
                                        ▼
                               DGS Types Dictionary
                                        │
                                        ▼
                              Dictionary Sign Pose
                │                       │
                └───────────┬───────────┘
                            │
                            ▼
                   Sign Replacement
                            │
                            ▼
              Modified / Non-Fluent Pose
                            │
                            ▼
                   Temporal Processing
                            │
                            ▼
                      Neural Model
                            │
                            ▼
                 Motion Reconstruction
                            │
                            ▼
                Fluent Pose Sequence
```

---

# Training Data Generation

One of the central components of the project is automatically creating training examples from naturally fluent sign language data.

The system uses sentence-level data from the **DGS Corpus** together with dictionary-level sign representations from **DGS Types**.

For each sentence:

```text
Original Sentence

SIGN_A → SIGN_B → SIGN_C → SIGN_D → SIGN_E
```

one or more signs can be selected:

```text
SIGN_A → [SIGN_B] → SIGN_C → [SIGN_D] → SIGN_E
```

and replaced by their dictionary forms:

```text
SIGN_A → DICT_B → SIGN_C → DICT_D → SIGN_E
```

This produces an intentionally degraded or less fluent pose sequence.

The original natural sentence remains the target.

Therefore:

```text
INPUT
Dictionary-modified pose sequence

              ↓

MODEL

              ↓

TARGET
Original fluent sentence-level pose sequence
```

This creates supervised training data without requiring every modified sentence to be manually recorded.

---

# Dataset

The project uses:

### DGS Corpus

Sentence-level sign language data containing naturally produced signing and linguistic annotations.

Used as the source of fluent target pose sequences.

### DGS Types Dictionary

Dictionary-level sign representations used to replace signs inside sentence-level sequences.

The combination allows the project to simulate a realistic sign-generation scenario where isolated dictionary signs are inserted into a sentence.

---

# Dataset Preparation

Dataset creation is handled by:

```bash
python fluent_pose_synthesis/data/create_data.py \
  --corpus_dir pose_data/tfds_dgs \
  --dictionary_dir pose_data/tfds_dgs \
  --output_dir pose_data/output
```

The generated dataset follows a structure similar to:

```text
pose_data/
│
├── tfds_dgs/
│
└── output/
    │
    ├── train/
    │   ├── train_1_original.pose
    │   ├── train_1_updated.pose
    │   ├── train_1_metadata.json
    │   └── ...
    │
    ├── validation/
    │
    └── test/
```

Each example contains multiple components.

### `*_original.pose`

The original fluent sentence-level pose sequence.

### `*_updated.pose`

The modified sequence containing dictionary-form sign replacements.

### `*_metadata.json`

Metadata describing the generated training example.

This separation makes it possible to construct explicit:

```text
Input Pose → Target Pose
```

training pairs.

---

# Pose-Level Learning

The system operates on **pose sequences rather than raw RGB video**.

This is important because sign language communication is fundamentally dependent on structured human motion.

A pose representation can encode information associated with:

```text
Body
│
├── Upper-body motion
├── Shoulder movement
└── Arm trajectories

Hands
│
├── Hand position
├── Hand movement
└── Signing trajectory

Face / Head
│
├── Head behavior
└── Non-manual components

Time
│
├── Motion duration
├── Transition timing
└── Sequence continuity
```

Working at the pose level allows the learning problem to focus directly on signing motion rather than background appearance, lighting, clothing, or camera characteristics.

---

# Temporal Alignment

A major challenge is that dictionary signs and sentence-level signs may have different durations.

For example:

```text
Original Sentence Sign

|----------------------|

Dictionary Sign

|-----------|
```

A direct replacement would create temporal inconsistencies.

The replacement therefore needs to be temporally adapted:

```text
Dictionary Pose
      │
      ▼
Stretch / Compress
      │
      ▼
Target Duration
      │
      ▼
Sentence-Compatible Segment
```

The neural processing stage is designed around this problem of integrating corrected or dictionary-level motion into the temporal structure of the original sentence.

---

# Fluency Reconstruction

Simply matching duration is not sufficient.

Consider:

```text
Previous Sign
     │
     ▼
[ abrupt boundary ]
     │
Dictionary Sign
     │
[ abrupt boundary ]
     ▼
Next Sign
```

Even when the individual sign is correct, these boundaries can make the sequence appear unnatural.

The objective is therefore to learn:

```text
Previous Context
       +
Replacement Sign
       +
Following Context
       │
       ▼
Context-Aware Motion Refinement
       │
       ▼
Smooth Continuous Signing
```

The model aims to improve motion continuity without changing the intended linguistic content.

---

# Prosody and Intonation

Sign language fluency involves more than reproducing isolated lexical signs.

Natural signing contains sentence-level characteristics including:

- Timing
- Rhythm
- Sign duration
- Motion continuity
- Prosodic structure
- Transitions between signs
- Context-dependent movement

This project therefore approaches sign synthesis as a **sequence-level motion problem**, rather than treating signs as independent visual units.

---

# Connection to Sign Language Generation

The system can operate as a post-processing stage after existing sign language generation methods.

For example:

```text
Text
 │
 ▼
Spoken-to-Signed Translation
 │
 ▼
Gloss Sequence
 │
 ▼
Dictionary Sign Retrieval
 │
 ▼
Initial Pose Sequence
 │
 ▼
Fluent Pose Synthesis
 │
 ▼
Refined Pose Sequence
 │
 ▼
Avatar / Video Generation
```

This means the project does not need to replace the entire translation or animation pipeline.

Instead, it targets one specific and difficult component:

> transforming mechanically assembled signing into more natural continuous motion.

---

# HamNoSys Integration Context

Generated sign language can also originate from symbolic sign representations such as **HamNoSys**.

A pipeline may look like:

```text
Gloss
  │
  ▼
HamNoSys
  │
  ▼
Pose Generation
  │
  ▼
Initial Signing
  │
  ▼
Fluency Refinement
```

This makes pose-level fluency correction potentially applicable to multiple upstream sign-generation systems.

---

# Environment Setup

Using Conda is recommended for dependency management.

### Clone the Repository

```bash
git clone https://github.com/sign-language-processing/fluent-pose-synthesis.git

cd fluent-pose-synthesis
```

### Create the Environment

```bash
conda env create -f environment.yml
```

### Activate It

```bash
conda activate fluent-pose
```

Alternatively, the package can be installed directly from GitHub:

```bash
pip install git+https://github.com/sign-language-processing/fluent-pose-synthesis
```

---

# Quick Debug Training

A lightweight debug configuration is available to verify the complete training pipeline.

Run:

```bash
python fluent_pose_synthesis/train.py \
  --name debug \
  --data pose_data/output \
  --save save/debug_run
```

The debug configuration:

```text
Training Samples: 16
Batch Size:       16
Epochs:           100
Output Directory: save/debug_run
```

This mode is useful for validating:

- Dataset loading
- Pose parsing
- Batch construction
- Model execution
- Training loop
- Loss computation
- Checkpoint generation
- Logging
- End-to-end pipeline integrity

before launching larger experiments.

---

# Training Workflow

The complete training process can be summarized as:

```text
DGS Sentence Data
       │
       ▼
Original Fluent Poses
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
Gloss Selection        DGS Dictionary
                              │
                              ▼
                     Dictionary Pose
                              │
       ┌──────────────────────┘
       ▼
Replace Selected Signs
       │
       ▼
Modified Pose Sequence
       │
       ▼
Temporal Processing
       │
       ▼
Neural Network
       │
       ▼
Predicted Fluent Sequence
       │
       ▼
Compare Against Original
       │
       ▼
Optimization
```

---

# Evaluation Strategy

The system is designed to be evaluated using both **objective and subjective criteria**.

### Objective Evaluation

Possible objective evaluation focuses on:

- Pose reconstruction quality
- Sign recognition accuracy
- Motion consistency
- Sequence quality

### Subjective Evaluation

Human evaluation can assess characteristics that are difficult to capture using numerical metrics alone:

- Naturalness
- Fluency
- Transition quality
- Perceived signing quality

This is particularly important for sign language synthesis because low numerical pose error does not necessarily guarantee natural signing.

---

# Engineering Challenges

The project addresses several difficult problems simultaneously.

### 1. Isolated-to-Continuous Signing

Dictionary signs are produced independently, while natural language consists of continuous contextual motion.

### 2. Variable Sequence Length

The same sign may have different durations depending on its context.

### 3. Temporal Alignment

Replacement signs need to be stretched or compressed while preserving meaningful motion.

### 4. Transition Modeling

Motion before and after a replacement must remain smooth.

### 5. Linguistic Preservation

Improving fluency must not unintentionally change the sign being communicated.

### 6. High-Dimensional Pose Sequences

Human pose sequences contain spatial and temporal dependencies that must be modeled jointly.

### 7. Automatic Training Pair Generation

The data pipeline must construct meaningful non-fluent/fluent sequence pairs from corpus and dictionary data.

---

# Repository Structure

```text
fluent-pose-synthesis/
│
├── fluent_pose_synthesis/
│   │
│   ├── data/
│   │   └── create_data.py
│   │
│   ├── train.py
│   │
│   └── ...
│
├── pose_data/
│   │
│   ├── tfds_dgs/
│   │
│   └── output/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── save/
│   └── debug_run/
│
├── environment.yml
│
└── README.md
```

---

# Research Motivation

Sign language generation is not only a translation problem.

A system may successfully determine the correct sequence of signs while still producing signing that looks mechanical or unnatural.

This creates a distinction between:

```text
Linguistically Correct
        ≠
Visually Fluent
```

A practical sign language generation system needs both.

This project explores the second problem by treating fluency as a learnable transformation over pose sequences.

---

# Key Contributions

The project provides an end-to-end framework for:

- Sign language pose post-editing
- Sentence-level pose processing
- Dictionary-to-sentence sign replacement
- Automatic paired-data generation
- Temporal sign alignment
- Pose-sequence neural modeling
- Motion fluency reconstruction
- Prosody-aware sign synthesis research
- DGS corpus integration
- DGS Types dictionary integration
- Training and checkpoint pipelines
- Objective and subjective evaluation design

---

# Technical Skills Demonstrated

This project demonstrates practical work across:

```text
Sign Language Processing
Pose Estimation & Representation
Computer Vision
Deep Learning
Sequence Modeling
Motion Synthesis
Temporal Modeling
Pose Sequence Processing
Dataset Engineering
Neural Network Training
Human Motion Analysis
Research Engineering
```

---

# Applications

Potential applications include:

### Sign Language Translation

Improving the motion quality of automatically generated translated signing.

### Sign Language Avatars

Generating smoother pose sequences before avatar animation.

### Sign Language Video Generation

Post-processing generated signing before rendering realistic video.

### Sign Language Editing

Replacing incorrect signs without regenerating an entire sentence.

### Accessibility Technology

Supporting more natural AI-generated sign language communication.

### Sign Language Research

Studying the relationship between isolated lexical signs and fluent sentence-level motion.

---

# Future Directions

The framework can be extended toward:

- Stronger temporal sequence models
- Diffusion-based motion refinement
- Improved sign-boundary modeling
- Explicit coarticulation modeling
- Facial and non-manual feature refinement
- Context-aware sign duration prediction
- Automatic sign spotting
- Larger-scale corpus training
- Sign-language-specific perceptual metrics
- Native signer evaluation
- Real-time pose correction
- Avatar integration
- End-to-end text-to-fluent-sign pipelines

---

# Project Vision

The long-term goal is to move sign language generation away from:

```text
Correct Signs + Mechanical Concatenation
```

toward:

```text
Correct Signs
      +
Natural Timing
      +
Continuous Motion
      +
Contextual Transitions
      +
Prosody
      ↓
Fluent Generated Signing
```

The project treats **fluency as a first-class modeling problem** rather than an afterthought of sign generation.

---

# Author

**Habeba Mohamed Fetouh**

AI / Machine Learning & Computer Vision Engineer

Research and development interests:

`Computer Vision` • `Deep Learning` • `Sign Language AI` • `Pose Processing` • `Human Motion Analysis` • `Sequence Modeling`

---

## Acknowledgements

This project builds on sign language resources and research infrastructure including the **DGS Corpus**, **DGS Types**, pose-based sign language representations, and existing sign language generation research.

---

## Disclaimer

This repository is intended for research and development in sign language processing and pose synthesis.

Sign language generation systems should be evaluated with appropriate linguistic expertise and, where possible, with members of the relevant Deaf and signing communities.

---

<p align="center">
  <b>From isolated signs to continuous, fluent motion.</b>
</p>
