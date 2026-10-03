import os
import urllib.request
import zipfile

def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "tiny-imagenet-200.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print("Extraction complete.")
    
    # Reformat validation set to match PyTorch ImageFolder structure
    val_dir = os.path.join(dest_dir, 'tiny-imagenet-200', 'val')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    val_images_dir = os.path.join(val_dir, 'images')
    
    if os.path.exists(val_annotations_file):
        print("Formatting validation directory for PyTorch ImageFolder...")
        with open(val_annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.split('\t')
                img_name = parts[0]
                class_name = parts[1]
                
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                src_path = os.path.join(val_images_dir, img_name)
                dst_path = os.path.join(class_dir, img_name)
                
                if os.path.exists(src_path):
                    os.rename(src_path, dst_path)
        print("Formatting complete.")

if __name__ == "__main__":
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dest_dir = "./data"
    
    if not os.path.exists(os.path.join(dest_dir, 'tiny-imagenet-200')):
        download_and_extract(url, dest_dir)
    else:
        print("Tiny-ImageNet already exists.")
