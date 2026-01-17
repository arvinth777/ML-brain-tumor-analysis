# 🚀 Deployment Guide - Brain Tumor Detection App

## Quick Start - Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Step-by-Step Deployment

#### 1. Push to GitHub

```bash
cd /Users/arvinthcinmayankirupakaran/mlbraintumordemo/ML-brain-tumor-analysis

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Add brain tumor detection Streamlit app"

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/arvinth777/ML-brain-tumor-analysis.git

# Push to GitHub
git push -u origin main
```

#### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `arvinth777/ML-brain-tumor-analysis`
4. Set main file path: `app.py`
5. Click "Deploy!"

#### 3. Update README

Once deployed, update the README.md with your live app URL:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-actual-app-url.streamlit.app)

**[Try the app here →](https://your-actual-app-url.streamlit.app)**
```

## Local Testing

### Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
# Make sure virtual environment is activated
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Test Checklist

- [ ] App loads without errors
- [ ] Can upload an image (JPG/PNG)
- [ ] Image preview displays correctly
- [ ] "Analyze Image" button works
- [ ] Prediction results appear
- [ ] Confidence score displays
- [ ] Preprocessed image shows
- [ ] Sidebar information is visible
- [ ] UI is responsive

## Troubleshooting

### Common Issues

**1. TensorFlow Installation Issues**
```bash
# If TensorFlow fails to install, try:
pip install tensorflow-macos  # For Mac M1/M2
# or
pip install tensorflow-cpu  # For CPU-only systems
```

**2. OpenCV Issues**
```bash
# Use headless version for deployment
pip install opencv-python-headless
```

**3. Memory Issues on Streamlit Cloud**
- The free tier has 1GB RAM limit
- ResNet50 model is ~100MB
- Should work fine for single users
- For high traffic, consider upgrading

### Streamlit Cloud Configuration

The `.streamlit/config.toml` file is already configured with:
- Dark theme
- Optimized settings for deployment
- Browser configuration

## Post-Deployment

### 1. Test the Live App
- Visit your deployed URL
- Test with sample MRI images
- Verify all features work

### 2. Update Documentation
- Add live demo link to README
- Add screenshots/GIFs
- Update badges

### 3. Monitor Usage
- Check Streamlit Cloud dashboard
- Monitor app performance
- Review user analytics

## Optional Enhancements

### Add Sample Images
Create a `samples/` directory with test images:
```bash
mkdir samples
# Add sample MRI images
```

Update `app.py` to include sample image selector.

### Add Model File
If you have a trained model:
```bash
mkdir models
# Add your trained .pkl or .h5 file
```

Update `app.py` to load the actual model instead of demo prediction.

### Custom Domain
- Go to Streamlit Cloud settings
- Add custom domain (requires paid plan)

## Security Notes

- Never commit API keys or secrets
- Use Streamlit secrets for sensitive data
- Keep model files under 100MB for free tier

## Support

For issues or questions:
- Open an issue on GitHub
- Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Streamlit community forum: [discuss.streamlit.io](https://discuss.streamlit.io)
