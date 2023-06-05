import wandb
import constants
import scipy.ndimage
from matplotlib import pyplot as plt
from PIL import Image
from custom_types import *
from custom_types import T, ARRAY, nnf, torch
from utils import files_utils, image_utils
import imageio
import os
import pathlib
import pickle

def is_grayscale(image: ARRAY) -> bool:
    if image.ndim == 2 or image.shape[-1] == 1:
        return True
    mask = np.equal(image[:, :, 0], image[:, :, 1]) * np.equal(image[:, :, 2], image[:, :, 1])
    return mask.all()


def crop_square(image:ARRAY) -> ARRAY:
    h, w, c = image.shape
    offset = abs(h - w)
    if h > w:
        image = image[offset // 2:  h - offset + offset // 2]
    elif h < w:
        image = image[:, offset // 2:  w - offset + offset // 2]
    return image


def resize(image_arr: ARRAY, max_edge_length: int) -> ARRAY:
    h, w, c = image_arr.shape
    max_edge = max(w, h)
    if max_edge < max_edge_length:
        return image_arr
    if c == 1:
        image_arr = image_arr[:, :, 0]
    image = Image.fromarray(image_arr)
    s = max_edge_length / float(max_edge)
    size = (int(w * s), int(h * s))
    image = V(image.resize(size, resample=Image.BICUBIC))
    if c == 1:
        image = np.expand_dims(image, 2)
    return image


def rba_to_rgb(path: str):

    rgba_image = Image.open(path)
    rgba_image.load()
    background = Image.new("RGB", rgba_image.size, (255, 255, 255))
    background.paste(rgba_image, mask = rgba_image.split()[3])
    return V(background)


def gifed(folder: str, interval: float, name: str, filter_by: Optional[Callable[[List[str]], bool]] = None,
          loop: int = 0, split: int = 1, reverse: bool = True, mp4=False, is_alpha: bool = False):
    folder = files_utils.add_suffix(folder, "/")
    files = files_utils.collect(folder, '.png')
    if filter_by is not None:
        files = list(filter(filter_by, files))
    files = sorted(files, key=lambda x: x[1])
    # files = sorted(files, key=lambda x: int(x[1].split('_L')[-1]))
    if len(files) > 0:
        if is_alpha:
            images = [[rba_to_rgb(''.join(file)) for file in files]]
        else:
            images = [[imageio.imread(''.join(file)) for file in files]]
        # images = [[np.transpose(image, (1,0,2)) for image in images[0]]]
        if split > 1:
            images_ = []
            for i, image in enumerate(images[0]):
                if i % split == 0:
                    images_.append([])
                images_[-1].append(image)
            images = images_
        for i, group in enumerate(images):
            if reverse:
                group_ = group.copy()
                group_.reverse()
                group = group + group_
                interval_ = interval
            else:
                interval_ = [interval] * len(group)
                interval_[0] = 1
                # interval_[-1] = 1.5
            extension = 'mp4' if mp4 else 'gif'
            if mp4:
                fps = (1. / interval)
                imageio.mimsave(f'{folder}{name}{str(i) if split > 1 else ""}.{extension}', group, fps=fps)
            else:
                imageio.mimsave(f'{folder}{name}{str(i) if split > 1 else ""}.{extension}',
                                group, duration=interval_, loop=loop)



def to_heatmap(vals: Union[T, ARRAY], palette: str = 'coolwarm') -> T:
    shape = vals.shape
    if type(vals) is T:
        vals: ARRAY = vals.detach().cpu().numpy()
    to_reshape = vals.ndim > 1
    if to_reshape:
        vals = vals.flatten()
    vals = (vals * 255).astype(np.uint8)
    colormap = plt.get_cmap(palette)
    np_heatmap = colormap(vals)[:, :3]
    # np_heatmap = np.ascontiguousarray(cv2.applyColorMap(np_vals, cv2.COLORMAP_HOT)[:, 0, ::-1])
    heatmap = torch.from_numpy(np_heatmap).float()
    if to_reshape:
        heatmap = heatmap.view(*shape, 3)
    return heatmap


def unroll_domain(h: int, w: int) -> T:
    vs_y = torch.linspace(-1, 1., h)
    vs_x = torch.linspace(-1, 1., w)
    vs = torch.meshgrid(vs_y, vs_x)
    vs = torch.stack(vs, dim=2)
    return vs

def random_sampling(image: ARRAY, scale: Union[float, int], non_uniform_sampling=False):
    h, w, c = image.shape
    coords = unroll_domain(h, w).view(-1, 2)
    labels = torch.from_numpy(image).reshape(-1, c).float() / 255
    masked_image = labels.clone()

    if scale < 1:
        split = int(h * w * scale)
    else:
        split = (h * w) // int(scale ** 2)

    if non_uniform_sampling:
        grayscale = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        edge_map = np.abs(scipy.ndimage.filters.laplace(grayscale))
        weight_map = edge_map / edge_map.sum()

        select = np.random.choice(a=h * w, size=split, replace=False, p=weight_map.reshape(-1))
        select = torch.from_numpy(select).long()
        masked = torch.ones(h*w)
        masked[select] = 0
        masked = torch.nonzero(masked).squeeze(-1)

        # Calculate probability for non-uniform sampling
        prob = torch.from_numpy(weight_map.reshape(-1)[select])
    else:
        indices = torch.arange(0, h * w)
        select = indices[indices % 2 == 0]
        masked = indices[indices % 2 == 1]

        # Calculate probability for uniform sampling
        prob = torch.ones((1))

    sample_cords = coords[select]
    sample_labels = labels[select]
    masked_image[masked] = 1
    masked_cords = coords[masked]
    masked_labels = labels[masked]
    masked_image = masked_image.view(h, w, c)
    return sample_labels, sample_cords, coords, masked_cords, masked_labels, masked_image, prob.float()



def grid_sampling(image: ARRAY, scale: int):
    h, w, c = image.shape
    coords = unroll_domain(h, w)
    labels = torch.from_numpy(image)[::scale, ::scale].reshape(-1, c).float() / 255
    # masked_image = labels.clone()
    sample_cords = (coords[::scale, ::scale]).reshape(-1, 2)
    return labels, sample_cords, coords.view(-1, 2), None


def init_source_target(path: Union[ARRAY, str], name: str, max_res: int, scale: Union[float, int],
                       square: bool = True, non_uniform_sampling=False):
    if isinstance(path, pathlib.PurePath):
        image = np.array(Image.open(path).convert('RGB'))
    else:
        image = path
    if is_grayscale(image):
        image = image[:, :, :1]
    if square:
        image = crop_square(image)
    image = resize(image, max_res)
    h, w, c = image.shape
    cache_path = constants.RAW_IMAGES / 'cache' / f'{name}_{scale}_{non_uniform_sampling}.pkl'

    if cache_path.exists():
        cache = files_utils.load_pickle(cache_path)
    else:
        cache = random_sampling(image, scale, non_uniform_sampling=non_uniform_sampling)
        files_utils.save_pickle(cache, cache_path)
    labels, samples, vs_base, masked_cords, mask_labels, masked_image, prob = cache
    image_labels = torch.from_numpy(image).reshape(-1, c).float() / 255
    return vs_base, samples, labels, image, image_labels, (masked_cords, mask_labels, masked_image), prob


def psnr(img1, img2, **_):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def plot_image(model, vs_in: T, ref_image: ARRAY, mask_model):
    model.eval()
    with torch.no_grad():
        if model.is_progressive or mask_model is not None:
            if mask_model is not None:
                _, mask = mask_model(vs_in)
                out = model(vs_in, override_mask=mask)
            else:
                out, mask = model(vs_in, get_mask=True)
            if mask.dim() != out.dim():
                mask: T = mask.unsqueeze(0).expand(out.shape[0], mask.shape[0])
            hm = torch.abs(mask[:, 2:]).sum(1) / torch.abs(mask[:, 2:]).sum(1).max()
            hm = image_utils.to_heatmap(hm)
            hm = hm.view(*ref_image.shape[:-1], 3)
        else:
            out = model(vs_in, get_mask=True)
            hm = None
        out = out.view(ref_image.shape)
    model.train()

    return out, hm


def log_evaluation_progress(model, image, out_path, tag, vs_base, device, mask_model=None, i = 0, labels = None):
    with torch.no_grad():
        extra = 'mask' if mask_model is not None else 'no_mask'
        out, hm = plot_image(
                    model, vs_base.to(device), image, mask_model)
        if labels is not None:
            wandb.log({'psnr eval': psnr(out.view(labels.shape), labels.to(device))})
        files_utils.export_image(
                    out, out_path / f'opt_{tag}_{extra}' / f'{i:04d}.png')
        wandb.log({'image': wandb.Image(str(out_path / f'opt_{tag}_{extra}' / f'{i:04d}.png'))})
        if hm is not None:
            files_utils.export_image(
                        hm, out_path / f'heatmap_{tag}_{extra}' / f'{i:04d}.png')
            wandb.log({f'heatmap': wandb.Image(str(out_path / f'heatmap_{tag}_{extra}' / f'{i:04d}.png'))})


def rgb_to_grayscale(img):
    # The weights correspond to the conversion formula: 
    # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, -1, 1, 1).to(img.device)
    return (img * weights).sum(dim=1, keepdim=True)


def ssim(img1, img2, img_shape=None, window_size=11, k1=0.01, k2=0.03, L=1.0, **_):
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = torch.ones((1, 1, window_size, window_size)) / (window_size ** 2)

    img1 = img1.view(1, img1.shape[1], img_shape[0], -1)
    img2 = img2.view(1, img2.shape[1], img_shape[0], -1)

    img1_gray = rgb_to_grayscale(img1)
    img2_gray = rgb_to_grayscale(img2)

    window = window.to(img1_gray.device)

    mu1 = nnf.conv2d(img1_gray, window, padding=window_size // 2, groups=1)
    mu2 = nnf.conv2d(img2_gray, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nnf.conv2d(img1_gray * img1_gray, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = nnf.conv2d(img2_gray * img2_gray, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = nnf.conv2d(img1_gray * img2_gray, window, padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return torch.mean(ssim_map)




