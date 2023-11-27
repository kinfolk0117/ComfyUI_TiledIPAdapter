import sys
import os

import torch
import comfy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

def generate_tiles(image_width, image_height, tile_width, tile_height, overlap, offset=0):
    tiles = []

    # first tile
    #tiles.append((0, 0, tile_width, tile_height))
    # set x and y for second tile based on offset
    #y = tile_height - overlap + offset
    y = 0
    while y < image_height:

        if y == 0:
            next_y = y + tile_height - overlap + offset
        else:
            next_y = y + tile_height - overlap

        if y + tile_height > image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        while x < image_width:
            if x == 0:
                next_x = x + tile_width - overlap + offset
            else:
                next_x = x + tile_width - overlap
            if x + tile_width > image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width


            tiles.append((x, y))

            if next_x > image_width:
                break
            x = next_x

        if next_y > image_height:
            break
        y = next_y

    return tiles

# Take one large image and split it into tiles, then process each tile individually
class TiledIPAdapterIntegrated:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

               "model": ("MODEL",),
                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                      "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                      "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                      "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                      "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                      "positive": ("CONDITIONING", ),
                      "negative": ("CONDITIONING", ),
                      "latent_image": ("LATENT",),
                      "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      "ipadapter": ("IPADAPTER", ),
                      "clip_vision": ("CLIP_VISION",),
                      "image": ("IMAGE", ),
                      "ipadapter_weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                      "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                      "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                      "overlap": ("INT", {"default": 64, "min": 1, "max": 10000}),
                      "offset": ("INT", {"default": 0, "min": 0, "max": 10000}),
                       }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "ipadapter"

    def process(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, ipadapter, clip_vision, image, ipadapter_weight , tile_width, tile_height, overlap, offset):
        from nodes import common_ksampler
        from ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterApply
        latent_image = latent_image['samples']
        # image.shape -> torch.Size([1, 1536, 1024, 3])
        image_height = image.shape[1]
        image_width = image.shape[2]

        latent_image_height = image_height // 8
        latent_image_width = image_width // 8
        latent_tile_height = tile_height // 8
        latent_tile_width = tile_width // 8

        tile_coordinates = generate_tiles(image_width, image_height, tile_width, tile_height, overlap, offset)

        iteration = 1
        blend = 64 // 8

        # for averaging we want to keep track of the number of samples per pixel, get size from latent_image
        count = torch.zeros((1, 1, latent_image_height, latent_image_width), dtype=latent_image.dtype, device=latent_image.device)
        
        for tile_coordinate in tile_coordinates:
            # convert to latent coordinates
            latent_tile_coordinate = (tile_coordinate[0] // 8, tile_coordinate[1] // 8)

            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))
            iteration += 1


            # Use image from original input as IPAdapter condition
            image_tile = image[:, tile_coordinate[1]:tile_coordinate[1]+tile_height, tile_coordinate[0]:tile_coordinate[0]+tile_width, :]

            from nodes import ImageScale
            image_tile = ImageScale.upscale(self, image_tile, 'nearest-exact', 512, 512, 'center')[0]
            patched_model = IPAdapterApply.apply_ipadapter(self, ipadapter, model, ipadapter_weight, clip_vision, image_tile,  "original", 0.0, None, None)[0]


            # Run ksampler on latent tile
            latent_tile = {'samples': latent_image[:, :, latent_tile_coordinate[1]:latent_tile_coordinate[1]+latent_tile_height, latent_tile_coordinate[0]:latent_tile_coordinate[0]+latent_tile_width]}

            channels = latent_tile['samples'].shape[1]
            weight_matrix = torch.ones((channels, latent_tile_height, latent_tile_width))

            # blend border
            for i in range(blend):
                weight = float(i) / blend
                weight_matrix[:, i, :] *= weight # top
                weight_matrix[:, -i-1, :] *= weight # bottom
                weight_matrix[:, :, i] *= weight # left
                weight_matrix[:, :, -i-1] *= weight # right




            old_tile_count = count[:, :, latent_tile_coordinate[1]:latent_tile_coordinate[1]+latent_tile_height, latent_tile_coordinate[0]:latent_tile_coordinate[0]+latent_tile_width]
            print(count.shape)
            print(latent_image.shape)
            print(latent_tile['samples'].shape)
            print(old_tile_count.shape)
            print(weight_matrix.shape)
            weight_matrix = weight_matrix * (old_tile_count != 0).float() + (old_tile_count == 0).float()

            new_latent_tile = common_ksampler(patched_model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_tile, denoise=denoise)[0]['samples']
            new_latent_tile = new_latent_tile * weight_matrix + latent_tile['samples'] * (1 - weight_matrix)

            # paste upscaled latent_tile tile into latent_image
            latent_image[:, :, latent_tile_coordinate[1]:latent_tile_coordinate[1]+latent_tile_height, latent_tile_coordinate[0]:latent_tile_coordinate[0]+latent_tile_width] = new_latent_tile
            count[:, :, latent_tile_coordinate[1]:latent_tile_coordinate[1]+latent_tile_height, latent_tile_coordinate[0]:latent_tile_coordinate[0]+latent_tile_width] = 1

        return ({'samples': latent_image}, )

# Expects already tile images
class TiledIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

               "model": ("MODEL",),
                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                      "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                      "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                      "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                      "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                      "positive": ("CONDITIONING", ),
                      "negative": ("CONDITIONING", ),
                      "latent_images": ("LATENT", ),
                      "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      "ipadapter": ("IPADAPTER", ),
                      "clip_vision": ("CLIP_VISION",),
                      "images": ("IMAGE", ),
                      "ipadapter_weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                       }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "ipadapter"

    def process(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_images, denoise, ipadapter, clip_vision, images, ipadapter_weight):
        from nodes import common_ksampler
        from ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterApply
        latents = []
        for i in range(latent_images['samples'].shape[0]):
            single_sample = latent_images['samples'][i:i+1]
            latent_image = {'samples': single_sample}
            image = images[i:i+1]

            foo = IPAdapterApply.apply_ipadapter(self, ipadapter, model, ipadapter_weight, clip_vision, image,  "original", 0.0, None, None)
            pathed_model = foo[0]
            latent = common_ksampler(pathed_model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            latents.append(latent[0]['samples'])
        return ({'samples': torch.cat(latents, dim=0)},)

NODE_CLASS_MAPPINGS = {
    "TiledIPAdapter": TiledIPAdapter,
    "TiledIPAdapterIntegrated": TiledIPAdapterIntegrated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledIPAdapter": "TiledIPAdapter",
    "TiledIPAdapterIntegrated": "TiledIPAdapterIntegrated",
}
