# TCAV Datasets 

Follow these links to download the datasets yourself.

---

## Step 1: Get Hugging Face Access

1. **Create account**: https://huggingface.co/join
2. **Get ImageNet access**: https://huggingface.co/datasets/ILSVRC/imagenet-1k
   - Click "Access repository"

3. **Get your token**: https://huggingface.co/settings/tokens
   - Create a new token and save it

---

## Step 2: Learn How to Download Datasets

**Official Hugging Face Documentation:**
- **Quickstart**: https://huggingface.co/docs/datasets/quickstart
- **Loading datasets**: https://huggingface.co/docs/datasets/loading
- **Authentication**: https://huggingface.co/docs/huggingface_hub/quick-start#authentication

---

## Step 3: Datasets You Need

### DTD (Textures Dataset)
- **Dataset page**: https://huggingface.co/datasets/cansa/Describable-Textures-Dataset-DTD
- **What you need**: Download "striped" and "dotted" texture images


### ImageNet (Zebra Images)
- **Dataset page**: https://huggingface.co/datasets/ILSVRC/imagenet-1k
- **What you need**: Filter for zebra class (label 340)

---

## Step 4: File Structure

Save images in this structure:
```
data/
├── broden/dtd/
│   ├── striped_0000.jpg
│   ├── striped_0001.jpg
│   ├── dotted_0000.jpg
│   └── dotted_0001.jpg
└── imagenet/zebra/
    ├── n02391049_train_0000.JPEG
    └── n02391049_val_0000.JPEG
```

