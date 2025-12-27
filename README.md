# Satellite Imagery-Based Property Valuation

A **Multimodal Regression Pipeline** that predicts property market value using both tabular data and satellite imagery. This project combines traditional housing features with visual environmental context extracted from satellite images to improve property valuation accuracy.

## 📋 Project Overview

This project develops a machine learning system that:
- Predicts property prices using tabular data (bedrooms, bathrooms, square footage, etc.)
- Enhances predictions by incorporating satellite imagery features
- Uses Convolutional Neural Networks (CNNs) to extract visual embeddings from satellite images
- Combines both data modalities through ensemble models (XGBoost + ExtraTrees)

## 🎯 Objectives

- Build a multimodal regression model to predict property value
- Programmatically acquire satellite imagery using latitude/longitude coordinates
- Perform exploratory and geospatial analysis
- Engineer features using CNNs to extract visual embeddings
- Test and compare fusion architectures (tabular vs multimodal)
- Ensure model explainability

## 📁 Project Structure

```
m_cdc/
├── data/
│   ├── train(1)(train(1)).csv      # Training dataset
│   └── test2.xlsx                  # Test dataset
├── property_images_v2/             # Satellite images for training properties
├── test_im/                         # Satellite images for test properties
├── main.ipynb                      # Main project notebook (DO NOT MODIFY)
├── preprocessing.ipynb             # Data preprocessing, EDA, and feature engineering
├── comparison.ipynb                # Model comparison: Tabular vs Multimodal
├── model_training.ipynb            # Model training pipeline
├── data_fetcher.py                 # Script to download satellite images
└── README.md                       # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Anaconda or Miniconda (recommended)

### Installation

1. **Clone the repository** (or download the project files)

2. **Create a conda environment** (recommended):
```bash
conda create -n property_valuation python=3.9
conda activate property_valuation
```

3. **Install required packages**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch torchvision pillow tqdm openpyxl requests
```

Or install from requirements.txt (if provided):
```bash
pip install -r requirements.txt
```

### Data Setup

1. **Place your data files** in the `data/` directory:
   - `train(1)(train(1)).csv` - Training dataset
   - `test2.xlsx` - Test dataset

2. **Download satellite images** (if not already downloaded):
```bash
python data_fetcher.py
```

   This will download satellite images for all properties in the training dataset and save them to `property_images/` folder.

   **Note:** The script uses Google Maps tile server which doesn't require an API key, but has rate limits. For large datasets, the download may take several hours.

## 📊 Usage

### 1. Data Preprocessing and EDA

Run `preprocessing.ipynb` to:
- Load and explore the dataset
- Perform comprehensive exploratory data analysis (EDA)
- Clean and preprocess the data
- Engineer features (log transformations, spatial features)
- Visualize correlations and distributions

### 2. Model Comparison

Run `comparison.ipynb` to:
- Train a tabular-only model (using only structured features)
- Train a multimodal model (tabular + satellite imagery)
- Compare performance metrics (R², MAE, RMSE)
- Visualize predictions and residuals

### 3. Model Training

Run `model_training.ipynb` to:
- Train the final multimodal model on the full dataset
- Generate predictions for the test set
- Save the model and predictions

### 4. Main Notebook

`main.ipynb` contains the complete pipeline including:
- Image downloading
- Data preprocessing
- Model training
- Test predictions

**⚠️ Important:** Do not modify `main.ipynb` as per project requirements.

## 🔧 Configuration

### Data Fetcher Configuration

Edit `data_fetcher.py` to customize:
- `OUTPUT_FOLDER`: Where to save downloaded images
- `ZOOM_LEVEL`: Image detail level (19 recommended)
- `IMAGE_SIZE`: Output image dimensions (512x512 standard)
- `MAX_WORKERS`: Number of parallel downloads

### Model Configuration

Model hyperparameters can be adjusted in the respective notebooks:
- XGBoost: `n_estimators`, `learning_rate`, `max_depth`, etc.
- ExtraTrees: `n_estimators`, `max_depth`, etc.
- Ensemble: VotingRegressor weights

## 📈 Model Architecture

### Tabular Model
- **Features:** 12 tabular features (bedrooms, bathrooms, sqft_living, etc.)
- **Model:** VotingRegressor (XGBoost + ExtraTrees)
- **Performance:** R² ≈ 0.877, MAE ≈ $74,000

### Multimodal Model
- **Features:** 13 features (12 tabular + 1 visual score)
- **Visual Feature Extraction:** ResNet18 CNN (pre-trained on ImageNet)
- **Model:** VotingRegressor (XGBoost + ExtraTrees)
- **Performance:** R² ≈ 0.879, MAE ≈ $71,000

### Feature Engineering
- **Log Transformation:** Applied to skewed features (price, sqft_living, etc.)
- **Spatial Features:** Distance from luxury hub (top 10% most expensive properties)
- **Visual Features:** Mean activation from ResNet18 feature extractor

## 📊 Results

### Performance Comparison

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Tabular Only | 0.8773 | $74,307 | ~$110,000 |
| Multimodal | 0.8785 | $71,437 | ~$108,000 |

**Improvement:** The multimodal approach shows a **~4% reduction in MAE** by incorporating satellite imagery features.

## 📝 Dataset Description

### Key Features

- **`price`**: Target variable (property price in USD)
- **`sqft_living`**: Total interior living space
- **`sqft_above`**: Interior space above ground level
- **`sqft_basement`**: Interior space below ground level
- **`sqft_lot`**: Total land area (lot size)
- **`sqft_living15` & `sqft_lot15`**: Average living/lot sizes of nearest 15 neighbors
- **`condition`** (1-5): Property maintenance quality
- **`grade`** (1-13): Construction quality and design
- **`view`** (0-4): View quality rating
- **`waterfront`**: Binary indicator (0/1)
- **`lat`, `long`**: Geographic coordinates

## 🔍 Key Insights

1. **Spatial Patterns:** Properties near the "luxury hub" (top 10% most expensive) command higher prices
2. **Visual Context:** Satellite imagery captures environmental factors (green cover, road density) that influence property value
3. **Feature Importance:** `sqft_living`, `grade`, and `lat` are the strongest predictors
4. **Multimodal Benefit:** Combining visual and tabular data improves prediction accuracy

## 🛠️ Troubleshooting

### Image Download Issues
- **Rate Limiting:** If downloads fail, reduce `MAX_WORKERS` in `data_fetcher.py`
- **Missing Images:** Some properties may not have valid coordinates; the script handles this gracefully

### Memory Issues
- **Large Datasets:** For datasets >20k properties, consider processing in batches
- **Image Processing:** Reduce `IMAGE_SIZE` if running out of memory

### Model Training Issues
- **CUDA Out of Memory:** Reduce batch size or use CPU (slower but works)
- **Convergence:** Adjust learning rate or number of estimators

## 📚 Dependencies

- **Data Handling:** pandas, numpy
- **Deep Learning:** torch, torchvision
- **Machine Learning:** scikit-learn, xgboost
- **Image Processing:** PIL (Pillow), opencv-python
- **Visualization:** matplotlib, seaborn
- **Utilities:** tqdm, requests

## 🤝 Contributing

This is a project submission. For questions or issues, please refer to the project documentation.

## 📄 License

This project is for educational purposes.

## 👤 Author

Data Science Project - Property Valuation with Multimodal Learning

---

**Note:** This project uses Google Maps tile server for satellite imagery. For production use, consider using official APIs (Google Maps Static API, Mapbox, Sentinel Hub) with proper API keys and rate limiting.

