import torch
from diffusers import LTXImageToVideoPipeline

MODEL_ID = "Lightricks/LTX-Video"
OUT_PATH = "portrait_prompt_embeds.pt"
DTYPE = torch.bfloat16

prompt = (
    "A close-up cinematic portrait of a person, natural face, expressive eyes, "
    "soft natural expression, hair gently moving in the wind, subtle blinking, "
    "slight head movement, shallow depth of field, soft background lights, "
    "camera slowly pushes in, realistic smooth video, natural skin texture, cinematic motion"
)

negative_prompt = (
    "static image, still frame, frozen video, no motion, distorted face, warped eyes, "
    "asymmetric eyes, deformed hair, unnatural smile, flickering face, identity shift, "
    "bad anatomy, blurry, low quality, watermark, text artifacts"
)

pipe = LTXImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
).to("cuda")

with torch.inference_mode():
    encoded = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=128,
        device=torch.device("cuda"),
        dtype=DTYPE,
    )

prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encoded

torch.save(
    {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "prompt_attention_mask": prompt_attention_mask.detach().cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.detach().cpu(),
        "negative_prompt_attention_mask": negative_prompt_attention_mask.detach().cpu(),
    },
    OUT_PATH,
)
