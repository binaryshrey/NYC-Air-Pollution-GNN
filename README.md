# NYC Urban Air Pollution Spatial Prediction Using GNN [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/binaryshrey/NYC-Air-Pollution-GNN/blob/main/NYC_air_pollution_gnn.ipynb)


## Project Overview

This project builds a real-world spatial machine learning pipeline to model and predict urban air pollution distribution using Graph Neural Networks (GNNs), real EPA monitoring data, and OpenStreetMap road networks.

The system integrates:

- EPA PM2.5 Air Pollution Monitoring Data
- NYC Road Network (OpenStreetMap via OSMnx)
- GIS Spatial Processing (GeoPandas, Spatial Joins)
- Graph Neural Networks (PyTorch Geometric)

## Problem Statement

Urban air pollution varies across neighborhoods due to traffic patterns, infrastructure density, and environmental conditions. Monitoring stations are sparse, so we need models that can predict pollution levels at unmeasured locations.

This project predicts: Pollution risk across NYC road network nodes using spatial graph learning.

## Methodology

```
    EPA PM2.5 Monitoring Data (CSV)
            ↓
    Filter NYC Counties
            ↓
    Convert to GeoDataFrame (Spatial Points)
            ↓
    Download NYC Road Network (OSMnx)
            ↓
    Spatial Join (Pollution → Road Nodes)
            ↓
    Construct Infrastructure Graph
            ↓
    Train Graph Neural Network
            ↓
    Predict Pollution Risk Distribution
```

## Data Sources

### Air Pollution Data

EPA AirData -- Daily PM2.5 (Parameter 88101)

Includes:

- Latitude / Longitude
- Daily PM2.5 Measurement
- Monitoring Station Metadata

### Infrastructure Data

OpenStreetMap (OSM) Power + Road Infrastructure

Extracted using:

- OSMnx Python Library
- Road Network Graph Construction

Data Attribution: © OpenStreetMap

## Spatial Data Processing

This project demonstrates real-world GIS processing:

- Vector Data Handling (GeoDataFrames)
- Coordinate Reference System Alignment
- Spatial Nearest Neighbor Joins
- Infrastructure Network Graph Construction

## Machine Learning Model

### Graph Neural Network Architecture

- Input Feature: PM2.5 Pollution Level
- Hidden Layers: Graph Convolution Layers
- Output: Pollution Risk Classification

Framework: - PyTorch

## Final Result Visualizations

### 1. Pollution Measurement Map

Shows raw EPA monitoring data across NYC.
![raw](https://raw.githubusercontent.com/binaryshrey/NYC-Air-Pollution-GNN/refs/heads/main/assets/image.png)

### 2. Pollution Distribution Histogram

Shows distribution of PM2.5 values across infrastructure nodes.
![PM2.5](https://raw.githubusercontent.com/binaryshrey/NYC-Air-Pollution-GNN/refs/heads/main/assets/image1.png)

### 3. GNN Predicted Pollution Risk Distribution

Shows predicted high-risk pollution zones across the NYC road network.
![rpredictedw](https://raw.githubusercontent.com/binaryshrey/NYC-Air-Pollution-GNN/refs/heads/main/assets/image2.png)

---

## Example Model Output Interpretation

Each node prediction:

    [ Probability Low Pollution , Probability High Pollution ]

Example:

    [0.82 , 0.18] → Low Risk Zone
    [0.35 , 0.65] → Elevated Pollution Risk

## Future Improvements

- Spatio-Temporal GNN Modeling
- Traffic + Population Exposure Integration
- Climate Scenario Air Quality Modeling
