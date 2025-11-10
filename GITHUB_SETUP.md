# GitHub Setup Guide

This guide will help you set up your GAN project on GitHub and deploy it to Hugging Face Spaces.

## Prerequisites

- GitHub account ([sign up here](https://github.com/join))
- Hugging Face account ([sign up here](https://huggingface.co/join))
- Git installed on your machine
- Python 3.9+ installed

## Step 1: Prepare Your Project

### 1.1 Generate Model Weights

**CRITICAL:** Before pushing to GitHub, make sure you've generated the model file:

1. Open `GAN_MNIST_Assignment.ipynb` in Jupyter
2. Run all cells from the beginning (Cells 1-26)
3. Cell 26 automatically saves `generator_model.pth` after training
4. Verify the file exists and is ~17 MB

```bash
# Check if model file exists
ls -lh generator_model.pth
```

### 1.2 Update Personal Information

Update these files with your information:

**All personal information has been updated:**
- GitHub username: vikranth1000
- Hugging Face username: rvikranth10
- LinkedIn: linkedin.com/in/vikranthreddimasu
- Repository name: mnist-gan

## Step 2: Initialize Git Repository

```bash
# Navigate to project directory
cd "/Users/vikranthreddimasu/Desktop/GAN Project"

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: MNIST GAN project with production-ready deployment"
```

## Step 3: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `mnist-gan` (or your preferred name)
3. Description: "Production-ready GAN for generating handwritten digits from MNIST dataset"
4. Visibility: Public (recommended for portfolio)
5. **DO NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

### Option B: Using GitHub CLI

```bash
gh repo create mnist-gan --public --description "Production-ready GAN for generating handwritten digits"
```

## Step 4: Push to GitHub

```bash
# Add remote repository
git remote add origin https://github.com/vikranth1000/mnist-gan.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 5: Update README with GitHub URL

Your repository URL is: `https://github.com/vikranth1000/mnist-gan`

## Step 6: Deploy to Hugging Face Spaces

### 6.1 Create Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Configure:
   - **Space name:** `mnist-gan` (or your preference)
   - **SDK:** Gradio
   - **Hardware:** CPU (free tier) or GPU (if you have credits)
   - **Visibility:** Public
3. Click "Create Space"

### 6.2 Upload Files

**Required Files:**
- `app.py`
- `requirements.txt`
- `README.md`
- `generator_model.pth` (the trained model - ~25-30 MB)

**Optional but Recommended:**
- `losses.png` (training visualization)
- `samples/epoch_*.png` (sample outputs)
- `LICENSE`

**Do NOT upload:**
- `data/` folder (too large, not needed)
- `.ipynb_checkpoints/`
- `.DS_Store`
- `GAN_MNIST_Assignment.ipynb` (optional - you can add it if you want)

### 6.3 Upload Methods

**Method 1: Web Interface**
1. Go to your Space page
2. Click "Files and versions" tab
3. Click "Add file" → "Upload files"
4. Drag and drop all required files
5. Wait for build (~3-5 minutes)

**Method 2: Git (Recommended)**
```bash
# Clone your Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/mnist-gan
cd mnist-gan

# Copy files from your project
cp "/Users/vikranthreddimasu/Desktop/GAN Project/app.py" .
cp "/Users/vikranthreddimasu/Desktop/GAN Project/requirements.txt" .
cp "/Users/vikranthreddimasu/Desktop/GAN Project/README.md" .
cp "/Users/vikranthreddimasu/Desktop/GAN Project/generator_model.pth" .
cp "/Users/vikranthreddimasu/Desktop/GAN Project/losses.png" .
cp -r "/Users/vikranthreddimasu/Desktop/GAN Project/samples" .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

## Step 7: Verify Deployment

1. Wait for build to complete (~3-5 minutes)
2. Visit your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/mnist-gan`
3. Test the application:
   - Generate digits with default settings
   - Try different seeds
   - Adjust temperature slider
   - Test example configurations

## Step 8: Update Links

After deployment, update these files:

**README.md:**
- Line 17: Update Hugging Face badge URL
- Line 29: Update demo link with your Space URL

**app.py:**
- Line 446: Update GitHub repository URL

## Step 9: Add to Portfolio/Resume

### GitHub Repository Features

Add these to make your repo stand out:

1. **Topics/Tags:** Add tags like `gan`, `pytorch`, `deep-learning`, `mnist`, `gradio`, `generative-models`
2. **About Section:** Add description and website link
3. **README Badges:** Already included, but verify they work

### Resume Entry

```
MNIST Digit Generator - Production GAN Application
PyTorch | Gradio | Deep Learning | MLOps

• Implemented Generative Adversarial Network from scratch with 1.49M parameters
• Achieved stable training over 200 epochs without mode collapse
• Developed production-grade application with comprehensive error handling and logging
• Deployed interactive demo on Hugging Face Spaces with real-time generation

Technologies: PyTorch, Gradio, NumPy, Python, Git
Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/mnist-gan
GitHub: https://github.com/YOUR_USERNAME/mnist-gan
```

## Troubleshooting

### Git Issues

**Problem:** "Repository not found"
- **Solution:** Check repository URL and ensure you have access

**Problem:** Large file size errors
- **Solution:** The model file (~25-30 MB) should be fine. If issues persist, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pth"
  git add .gitattributes
  ```

### Hugging Face Deployment Issues

**Problem:** Build fails
- **Solution:** Check build logs, verify `requirements.txt` has correct versions

**Problem:** Model not loading
- **Solution:** Ensure `generator_model.pth` was uploaded and is in root directory

**Problem:** Application errors
- **Solution:** Check logs in Space → Logs tab, verify all dependencies installed

## Next Steps

1. Push to GitHub
2. Deploy to Hugging Face
3. Update all links
4. Share on LinkedIn/Twitter
5. Add to portfolio
6. Update resume

## Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)

---

**Need Help?** Check the `DEPLOYMENT.md` file for more detailed deployment instructions.

