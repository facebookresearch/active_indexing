import argparse, os, json, random
from typing import Any, Dict, List, NamedTuple, Tuple

from PIL import Image
import numpy as np

import augly.utils as utils
import augly.image as imaugs


RNG = np.random.RandomState
rng = np.random.RandomState(0)
ParametersDistributions = NamedTuple

class ParameterDistribution:
    """Define how to sample a parameter"""

    def __init__(self, low: Any, high: Any):
        self.low = low
        self.high = high

    def sample(self, rng: RNG) -> Any:
        raise NotImplementedError()

class FixedVariable(ParameterDistribution):
    def __init__(self, value: Any):
        super().__init__(0, 0)
        self.value = value

    def sample(self, rng: RNG) -> Any:
        return self.value

class UniformFloat(ParameterDistribution):
    def sample(self, rng: RNG) -> float:
        return float(rng.uniform(self.low, self.high))

class UniformInt(ParameterDistribution):
    def sample(self, rng: RNG) -> int:
        return int(rng.randint(self.low, self.high + 1))

class UniformColor(ParameterDistribution):
    def sample(self, rng: RNG) -> Tuple[int, int, int]:
        return tuple(int(rng.randint(self.low, self.high)) for _ in range(3))

class UniformChoice(ParameterDistribution):
    def __init__(self, choices: List[Any]):
        super().__init__(0, 0)
        self.choices = choices

    def sample(self, rng: RNG) -> Any:
        if not self.choices:
            return None
        index = rng.randint(0, len(self.choices))
        return self.choices[index]

class UniformBool(ParameterDistribution):
    def __init__(self):
        super().__init__(0, 0)

    def sample(self, rng: RNG) -> bool:
        return bool(UniformInt(0, 1).sample(rng))

class TextChoice(ParameterDistribution):
    def sample(self, rng: RNG) -> List[int]:
        length = UniformInt(self.low, self.high).sample(rng)
        return [UniformInt(0, 10000).sample(rng) for _ in range(length)]

class ListPD(ParameterDistribution):
    def __init__(self, pds: List[ParameterDistribution]):
        super().__init__(0, 0)
        self.pds = pds

    def sample(self, rng: RNG) -> List[Any]:
        return [pd.sample(rng) for pd in self.pds]

class TuplePD(ParameterDistribution):
    def __init__(self, pds: List[ParameterDistribution]):
        super().__init__(0, 0)
        self.pds = pds

    def sample(self, rng: RNG) -> Tuple:
        return tuple(pd.sample(rng) for pd in self.pds)

class ExponentialInt(ParameterDistribution):
    def __init__(self, scale: float, low: int, high: int):
        super().__init__(low, high)
        self.scale = scale

    def sample(self, rng) -> int:
        # if we sample a value larger than `high`, we need to resample a new one
        # if we just take the min(x, high), it will change the distribution
        while True:
            r = rng.exponential(scale=self.scale)
            if int(r + self.low) <= self.high:
                return int(r + self.low)

class SymmetricFactor(ParameterDistribution):
    def sample(self, rng: RNG) -> float:
        factor = float(rng.uniform(self.low, self.high))
        invert = rng.randint(0, 2)
        return 1 / factor if invert else factor

class UniformLeftRightFactor(ParameterDistribution):
    def sample(self, rng: np.random.RandomState) -> Tuple[float, float]:
        width = float(rng.uniform(self.low, self.high))
        left = rng.uniform(0, 1 - width)
        right = left + width
        return left, right

class MediaFilterParameters(NamedTuple):
    """Contains the parameters to apply a video filter.
    This defines a unique and reproducible transformation"""

    name: str
    kwargs: Dict[str, Any]

    def __repr__(self) -> str:
        return json.dumps({**{"name": self.name}, **self.kwargs})

class MediaFilterWithPD(NamedTuple):
    """Define a filter and how to sample all its parameters"""

    # filter name, must match one the function method in this file
    name: str
    # must contains only ParameterDistribution attributes
    pd: ParametersDistributions

class AspectRatioPD(ParametersDistributions):
    ratio: UniformFloat = UniformFloat(0.5, 2.0)

class BlurPD(ParametersDistributions):
    radius: UniformFloat = UniformFloat(5.0, 10.0)

class BlurryMaskPD(ParametersDistributions):
    background_image: UniformChoice
    overlay_size: UniformFloat = UniformFloat(0.3, 0.8)
    x_pos: UniformFloat = UniformFloat(0, 1.0)
    y_pos: UniformFloat = UniformFloat(0, 1.0)

class BrightnessPD(ParametersDistributions):
    factor: UniformFloat = UniformFloat(0.1, 1.9)

class ClipImageSizePD(ParametersDistributions):
    min_resolution: UniformChoice = UniformChoice([500])
    max_resolution: UniformChoice = UniformChoice([3000000])

class ConvertColorPD(ParametersDistributions):
    mode: UniformChoice = UniformChoice(["P"])
    colors: UniformInt = UniformInt(2, 16)

class CropPD(ParametersDistributions):
    xs: UniformLeftRightFactor = UniformLeftRightFactor(0.3, 0.6)
    ys: UniformLeftRightFactor = UniformLeftRightFactor(0.3, 0.6)

class EncodingQualityPD(ParametersDistributions):
    quality: UniformInt = UniformInt(5, 25)

class EnhanceEdgesPD(ParametersDistributions):
    pass

class GrayscalePD(ParametersDistributions):
    pass

class HFlipPD(ParametersDistributions):
    pass

class IdentityPD(ParametersDistributions):
    pass

class OverlayEmojiPD(ParametersDistributions):
    emoji_path: UniformChoice
    x_pos: UniformFloat = UniformFloat(0.0, 0.8)
    y_pos: UniformFloat = UniformFloat(0.0, 0.8)
    opacity: UniformFloat = UniformFloat(0.5, 1.0)
    emoji_size: UniformFloat = UniformFloat(0.4, 0.8)

class OverlayOntoImagePD(ParametersDistributions):
    background_image: UniformChoice
    overlay_size: UniformFloat = UniformFloat(0.3, 0.6)
    x_pos: UniformFloat = UniformFloat(0, 0.4)
    y_pos: UniformFloat = UniformFloat(0, 0.4)

class OverlayOntoScreenshotPD(ParametersDistributions):
    template_filepath: UniformChoice
    crop_src_to_fit: UniformChoice = UniformChoice([True])


class OverlayTextPD(ParametersDistributions):
    font_file: UniformChoice
    text: TextChoice = TextChoice(5, 15)
    font_size: UniformFloat = UniformFloat(0.1, 0.3)
    color: UniformColor = UniformColor(0, 255)
    x_pos: UniformFloat = UniformFloat(0.0, 0.6)
    y_pos: UniformFloat = UniformFloat(0.0, 0.6)


class PadSquarePD(ParametersDistributions):
    color: UniformColor = UniformColor(0, 255)


class PerspectiveTransformPD(ParametersDistributions):
    sigma: UniformFloat = UniformFloat(30.0, 60.0)
    crop_out_black_border: UniformChoice = UniformChoice([True])


class PixelizationPD(ParametersDistributions):
    ratio: UniformFloat = UniformFloat(0.2, 0.5)


class RotatePD(ParametersDistributions):
    degrees: UniformFloat = UniformFloat(-90.0, 90.0)


class SaturationPD(ParametersDistributions):
    factor: UniformFloat = UniformFloat(2.0, 5.0)


class ShufflePixelsPD(ParametersDistributions):
    factor: UniformFloat = UniformFloat(0.1, 0.3)

def sample(rng: RNG, filter_with_pd: MediaFilterWithPD) -> MediaFilterParameters:
    """Sample for each ParameterDistribution attribute and
    return a dict with sampled parameters
    """
    kwargs = {key: pdi.sample(rng) for key, pdi in filter_with_pd.pd._asdict().items()}
    return MediaFilterParameters(name=filter_with_pd.name, kwargs=kwargs)

def sample_img_filters_parameters(rng: RNG, available_filters: List[MediaFilterWithPD]) -> List[MediaFilterParameters]:
    """Sample parameters for each available filters"""
    return [sample(rng, vf) for vf in available_filters]

def get_assets(emoji_dir: str, font_dir: str, screenshot_dir: str) -> Tuple[List[str], List[str], List[str]]:
    emojis = []
    for fn in utils.pathmgr.ls(emoji_dir):
        fp = os.path.join(emoji_dir, fn)
        if utils.pathmgr.isdir(fp):
            emojis.extend([os.path.join(fp, f) for f in utils.pathmgr.ls(fp)])

    fonts = [
        os.path.join(font_dir, fn)
        for fn in utils.pathmgr.ls(font_dir)
        if fn.endswith(".ttf")
    ]

    template_filenames = [
        os.path.join(screenshot_dir, fn)
        for fn in utils.pathmgr.ls(screenshot_dir)
        if fn.split(".")[-1] != "json"
    ]

    return emojis, fonts, template_filenames

emojis, fonts, template_filenames = get_assets(
        utils.EMOJI_DIR, utils.FONTS_DIR, utils.SCREENSHOT_TEMPLATES_DIR
    )

primitives = {
    "color": [
        MediaFilterWithPD(name="brightness", pd=BrightnessPD()),
        MediaFilterWithPD(name="grayscale", pd=GrayscalePD()),
        MediaFilterWithPD(name="saturation", pd=SaturationPD()),
    ],
    "overlay": [
        MediaFilterWithPD(
            name="overlay_emoji",
            pd=OverlayEmojiPD(emoji_path=UniformChoice(emojis)),
        ),
        MediaFilterWithPD(
            name="overlay_text", pd=OverlayTextPD(font_file=UniformChoice(fonts))
        ),
    ],
    "pixel-level": [
        MediaFilterWithPD(name="blur", pd=BlurPD()),
        MediaFilterWithPD(name="convert_color", pd=ConvertColorPD()),
        MediaFilterWithPD(name="encoding_quality", pd=EncodingQualityPD()),
        MediaFilterWithPD(name="apply_pil_filter", pd=EnhanceEdgesPD()),
        MediaFilterWithPD(name="pixelization", pd=PixelizationPD()),
        MediaFilterWithPD(name="shuffle_pixels", pd=ShufflePixelsPD()),
    ],
    "spatial": [
        MediaFilterWithPD(name="crop", pd=CropPD()),
        MediaFilterWithPD(name="hflip", pd=HFlipPD()),
        MediaFilterWithPD(name="change_aspect_ratio", pd=AspectRatioPD()),
        MediaFilterWithPD(
            name="overlay_onto_screenshot",
            pd=OverlayOntoScreenshotPD(
                template_filepath=UniformChoice(template_filenames)
            ),
        ),
        MediaFilterWithPD(name="pad_square", pd=PadSquarePD()),
        MediaFilterWithPD(
            name="perspective_transform", pd=PerspectiveTransformPD()
        ),
        MediaFilterWithPD(name="rotate", pd=RotatePD()),
    ],
}
post_filters = []

def augment_img(img, rng: RNG = rng, return_params=False):
    """
    Sample augmentation parameters for img.
    Args:
        img: query image.
    """

    # select filters to apply
    num_filters = rng.choice(np.arange(1, 5), p=[0.1, 0.2, 0.3, 0.4])
    filter_types_to_apply = rng.choice(
        np.asarray(list(primitives.keys())), size=num_filters, replace=False
    )
    filters_to_apply = [
        primitives[ftype][rng.randint(0, len(primitives[ftype]))]
        for ftype in filter_types_to_apply
    ]
    filters_to_apply += post_filters
    
    # Ensure that crop is in first position if selected and that convert_color is in last position if selected
    for j, vf in enumerate(filters_to_apply):
        if vf.name == "crop":
            filters_to_apply[j], filters_to_apply[0] = (
                filters_to_apply[0],
                filters_to_apply[j],
            )
        if vf.name == "convert_color":
            filters_to_apply[j], filters_to_apply[-1] = (
                filters_to_apply[-1],
                filters_to_apply[j],
            )

    # sample parameters for each filter
    all_filters_parameters = sample_img_filters_parameters(
        rng, filters_to_apply
    )

    # apply filters
    for j, ftr in enumerate(all_filters_parameters):
        aug_func = getattr(imaugs, ftr.name, None)
        kwargs = ftr.kwargs
        if ftr.name == "crop":
            x1, x2 = kwargs.pop("xs")
            y1, y2 = kwargs.pop("ys")
            kwargs["x1"], kwargs["x2"] = x1, x2
            kwargs["y1"], kwargs["y2"] = y1, y2
        img = aug_func(image=img, **kwargs)
    img = img.convert('RGB')
    if return_params:
        return img, all_filters_parameters
    else:
        return img


if __name__ == '__main__':

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str, default='output')
        parser.add_argument("--data_dir", type=str, default="/img/data/dir/")
        parser.add_argument("--seed", type=int, default=42)
        
        return parser

    params = get_parser().parse_args()
    print("__log__:{}".format(json.dumps(vars(params))))

    # set seed
    np.random.seed(params.seed)
    random.seed(params.seed)
    rng = np.random.RandomState(params.seed)

    # Load data
    print("Loading filenames from {}".format(params.data_dir))
    filenames = os.listdir(params.data_dir)

    # Generate augmented images
    print("Generating augmented images into {}".format(params.output_dir))
    augmentations = []
    os.makedirs(params.output_dir, exist_ok=True)
    for filename in filenames:
        img_path = os.path.join(params.data_dir, filename)
        img = Image.open(img_path)
        img, filters = augment_img(img, rng, return_params=True)
        img.convert('RGB').save(os.path.join(params.output_dir, filename), quality=95)
        augmentations.append(filters)
        print(filename, "[" + ", ".join([str(ftr) for ftr in filters]) + "]")
        # break

    # Save augmentations
    print("Saving augmentations")
    with open(os.path.join(params.output_dir, "augmentations.txt"), "a") as f:
        for augmentation in augmentations:
            line = "[" + ", ".join([str(ftr) for ftr in augmentation]) + "]\n"
            f.write(line)
        