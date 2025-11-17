# Melbourne Housing Price Prediction with PyTorch

A deep learning project predicting Melbourne house prices using PyTorch.

## 🎯 Results

- **Test Error:** $162,451 average prediction error
- **Model:** 4-layer neural network (22 → 128 → 64 → 32 → 1)
- **Features:** 11 numeric + 2 categorical (Type, Region)
- **Dataset:** 6,830 houses after cleaning

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
# Place in: data/raw/melb_data.csv

# Train model
cd src
python train.py
```

## 📊 Features Used

**Numeric (11):**
- Rooms, Bedrooms, Bathrooms, Car spaces
- Landsize, Building area, Year built
- Latitude, Longitude, Property count, Distance from CBD

**Categorical (2):**
- Property type (house/unit/townhouse)
- Region name (8 regions)

## 🏗️ Architecture
```
Input (22 features)
    ↓
Dense (128) + ReLU
    ↓
Dense (64) + ReLU
    ↓
Dense (32) + ReLU
    ↓
Output (1 - price prediction)
```

## 📈 Training

- **Loss:** Mean Absolute Error (MAE)
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 200 with batch_size=1
- **Train/Test Split:** 80/20

## 🛠️ Project Structure
```
melbourne-housing-pytorch/
├── src/
│   ├── simple_model.py    # Neural network architecture
│   └── train.py           # Training script
├── data/raw/              # Dataset (not in repo)
├── requirements.txt
└── README.md
```

## 📝 Learning Journey

This project was built to learn:
- PyTorch fundamentals (tensors, models, training loops)
- Neural network design decisions
- Feature engineering and preprocessing
- Train/test evaluation

**Key learnings:**
- Started with 5 features → $349k error
- Added deeper network → $200k error
- Added more features → $176k error  
- Added categorical encoding → $162k error (54% improvement!)

## 🔮 Future Improvements

- [ ] Add Suburb encoding (314 unique values)
- [ ] Implement mini-batch training optimization
- [ ] Add learning rate scheduling
- [ ] Create prediction visualization
- [ ] Deploy as simple web app

## 📄 License

MIT License - Free to use for learning purposes