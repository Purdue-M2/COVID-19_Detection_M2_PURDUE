import clip
import torch
from PIL import Image
from tqdm.auto import tqdm


def extract(image_paths, batch_size=128):

    device = 'cuda'
    model, preprocess = clip.load('ViT-L/14', device=device)

    features = []

    # "Batch process images"
    for i in tqdm(range(0, len(image_paths), batch_size), desc='extract'):
        batch_paths = image_paths[i:i + batch_size]

        array_size = len(batch_paths)
        batch_images = torch.zeros((array_size, 3, 224, 224), dtype=torch.float32, device=device)

        for j in range(array_size):
            # print('UniAttackData/phase1/' + batch_paths[j])
            image_path = batch_paths[j]
            # print(image_path)
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)
            batch_images[j] = image

        # Convert a list of images to a tensor
        batch_images_tensor = batch_images.to(device)

        # extract feature using clip
        with torch.no_grad():
            batch_features = model.encode_image(batch_images_tensor)

        features.append(batch_features.cpu())

    features = torch.cat(features, dim=0)

    return features

def extract_resnet(image_paths, batch_size=128):

    device = 'cuda'
    model, preprocess = clip.load('RN50', device=device)

    features = []

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc='extract'):
        batch_paths = image_paths[i:i + batch_size]

        array_size = len(batch_paths)
        batch_images = torch.zeros((array_size, 3, 224, 224), dtype=torch.float32, device=device)

        for j in range(array_size):
            # print('UniAttackData/phase1/' + batch_paths[j])
            image_path = batch_paths[j]
            # print(image_path)
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image)
            batch_images[j] = image

        batch_images_tensor = batch_images.to(device)

        # Extract features using the CLIP RN50 model
        with torch.no_grad():
            batch_features = model.encode_image(batch_images_tensor)

        features.append(batch_features.cpu())

    features = torch.cat(features, dim=0)

    return features


if __name__ == '__main__':

    pass
