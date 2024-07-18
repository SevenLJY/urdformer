import os, json
from tqdm import tqdm
from PIL import Image

def _resize(img_path, tgt_size=256):
    '''
    Resize the image to 256x256, center crop to 224x224, and make the background white.
    '''
    img = Image.open(img_path)
    img = img.resize(size=(tgt_size, tgt_size), resample=Image.BICUBIC)
    img = img.crop(box=(16, 16, 240, 240)) # center crop to 224x224
    img = _make_white_background(img)
    return img

def _make_white_background(src_img):
    src_img.load() # required for png.split()
    background = Image.new("RGB", src_img.size, (255, 255, 255))
    background.paste(src_img, mask=src_img.split()[3]) # 3 is the alpha channel
    return background

def load_model_ids(split_file, storage_only=False):
    with open(split_file, 'r') as f:
        data = json.load(f)
    if storage_only:
        test_ids = [model_id for model_id in data['test'] if 'Storage' in model_id]
    else:
        test_ids = data['test']
    return test_ids

if __name__ == '__main__':
    src_root = '/localhome/jla861/Documents/projects/im-gen-ao/data'
    dst_root = 'test_data/images'
    split_file = f'/localhome/jla861/Documents/projects/im-gen-ao/svr-ao/src/data/data_split.json'
    test_ids = load_model_ids(split_file)
    
    for model_id in tqdm(test_ids): # 76
        fname = model_id.replace('/', '_')

        src_path = os.path.join(src_root, model_id, 'imgs', '18.png')
        dst_path = os.path.join(dst_root, f'{fname}_18.png')
        img = _resize(src_path)
        img.save(dst_path)

        src_path = os.path.join(src_root, model_id, 'imgs', '19.png')
        dst_path = os.path.join(dst_root, f'{fname}_19.png')
        img = _resize(src_path)
        img.save(dst_path)