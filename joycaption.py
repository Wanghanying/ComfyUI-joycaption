import os
import torch


from loguru import logger
from torchvision import transforms
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

CLIP_PATH = os.getenv(
    "JOY_CAPTION_CLIP_PATH",
    "/share_nfs/hf_models/google-siglip-so400m-patch14-384",
)
VLM_PROMPT = "A beautiful descriptive caption for this image:\n"
MODEL_PATH = os.getenv(
    "JOY_CAPTION_MODEL_PATH",
    "/share_nfs/hf_models/meta-llama/Meta-Llama-3.1-8B-Instruct",
)
ADAPTER_MODEL_PATH = os.getenv(
    "JOY_CAPTION_ADAPTER_MODEL_PATH",
    "/share_nfs/hf_models/fancyfeast/joy-caption-pre-alpha",
)

class ImageAdapter(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_features, output_features)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class LoadJoyCaption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            }
        }
    RETURN_TYPES = ("JoyCaptionMODEL",)
    RETURN_NAMES = ("models",)
    FUNCTION = "loadmodel"
    CATEGORY = "JoyCaption"

    def loadmodel(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
        clip_model = AutoModel.from_pretrained(CLIP_PATH)
        clip_model = clip_model.vision_model
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
            tokenizer, PreTrainedTokenizerFast
        ), f"Tokenizer is of type {type(tokenizer)}"
        
        text_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map=device, torch_dtype=torch.bfloat16
        )
        text_model.eval()
        image_adapter = ImageAdapter(
            clip_model.config.hidden_size, text_model.config.hidden_size
        )
        image_adapter.load_state_dict(
            torch.load(f"{ADAPTER_MODEL_PATH}/image_adapter.pt", map_location="cpu")
        )
        image_adapter.eval()
        image_adapter.to(device)

        JoyCaptionMODEL = (clip_processor,tokenizer,clip_model, image_adapter,text_model) 
        return (JoyCaptionMODEL,)


class JoyImageCaption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "models": ("JoyCaptionMODEL",),
                "do_sample": ("BOOLEAN", {"default": True}),
                "max_new_tokens": ("INT", {"default": 300, "min": 1, "max": 0xfffffff}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT",{"default": 10}),
                "suppress_tokens": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),               
            }
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES =("text",) 

    FUNCTION = "imagecaption"
    CATEGORY = "JoyCaption"

    def imagecaption(self, image, do_sample, max_new_tokens,temperature, top_k, suppress_tokens, models):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        SIZE_LIMIT = 1536
        
        logger.debug(f"{do_sample=}, {max_new_tokens=}, {temperature=}")       
        # suppress_tokens = [part.strip() for part in suppress_tokens.split(',')]
        clip_processor,tokenizer,clip_model, image_adapter,text_model = models

        if len(image.shape) == 4:
            b, height, width, _ = image.shape  # b, h, w, c in ComfyUI
            if b != 1:
                error_msg = f"Image batch size should be 1, got {b}"
                logger.warning(error_msg)
                return {
                    "type": "bizyair",
                    "error": error_msg,
                }
            image = image.squeeze(0)
            image = image.permute(2, 0, 1)
        elif len(image.shape) == 3:  # c, h, w in PIL, for test python script
            _, height, width = image.shape
        if not (width <= SIZE_LIMIT and height <= SIZE_LIMIT):
            error_msg = f"Image size should be less than {SIZE_LIMIT}x{SIZE_LIMIT}, got {width}x{height}"
            logger.warning(error_msg)
            return {
                "type": "bizyair",
                "error": error_msg,
            }
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(image)

        image = clip_processor(images=pil_img, return_tensors="pt").pixel_values
        image = image.to(device)

        # Tokenize the prompt
        prompt = tokenizer.encode(
            VLM_PROMPT,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        # Embed image
        with torch.amp.autocast_mode.autocast(device, enabled=True):
            vision_outputs = clip_model(
                pixel_values=image, output_hidden_states=True
            )
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to(device)

        # Embed prompt
        prompt_embeds = text_model.model.embed_tokens(prompt.to(device))
        assert prompt_embeds.shape == (
            1,
            prompt.shape[1],
            text_model.config.hidden_size,
        ), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], self.text_model.config.hidden_size)}"

        embedded_bos = text_model.model.embed_tokens(
            torch.tensor(
                [[tokenizer.bos_token_id]],
                device=text_model.device,
                dtype=torch.int64,
            )
        )

        # Construct prompts
        inputs_embeds = torch.cat(
            [
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            ],
            dim=1,
        )

        input_ids = torch.cat(
            [
                torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                prompt,
            ],
            dim=1,
        ).to(device)
        attention_mask = torch.ones_like(input_ids)

        suppress_tokens = tokenizer.encode(suppress_tokens, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        suppress_tokens=[aa.tolist() for aa in suppress_tokens]
        
        generate_ids = text_model.generate(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k = top_k,
            temperature=temperature,
            suppress_tokens=suppress_tokens,
        )

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1] :]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        return (caption.strip(),)


NODE_CLASS_MAPPINGS = {
    "LoadJoyCaptionModel": LoadJoyCaption,
    "JoyImageCaption": JoyImageCaption,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadJoyCaptionModel": "LoadJoyCaption",
    "JoyImageCaption": "ImageJoyCaption",
}