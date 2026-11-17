# Data - AI4Mars (NASA JPL)
1. Kaggle: Search 'AI4Mars' -> Download train + test zips (18,100 train + 322 expert test)
   Or: https://www.kaggle.com/datasets - AI4Mars
   Or NASA: https://ai4mars.jpl.nasa.gov/
2. Unzip to:
data/ai4mars/train/images/*.jpg
data/ai4mars/train/labels/*.png
data/ai4mars/test/images/*.jpg
data/ai4mars/test/labels/*.png
3. Labels: 0=soil, 1=bedrock, 2=sand, 3=big rock, 255=unlabeled/rover/range>30m - IGNORED
4. For LOCAL test (Mac CPU): download only ~200 images, use --limit 200
5. For FULL train: use Colab T4 GPU, see train_colab.ipynb
