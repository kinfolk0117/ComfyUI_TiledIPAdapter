import sys
import os

import torch
import comfy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledIPAdapter": "TiledIPAdapter",
}
