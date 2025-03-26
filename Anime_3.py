import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# 选择模型（本地运行可能更适合 NSFW）
model_name = "hakurei/waifu-diffusion"

# 加载模型（启用 FP16 加速）
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# 使用 DDIM 采样（加快生成）
pipe.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

# 设定 Prompt（确保全身 + 隐藏手）
prompt = (
    "1girl, white hair, purple eyes, anime style, witch hat, highly detailed, masterpiece, best quality, cute, fantasy background, soft lighting"
)


# 负面 Prompt（避免错误生成）
negative_prompt = (
    "multiple people, blurry, cropped, bad anatomy, extra limbs, bad hands, bad fingers, "
    "clothes, dress, jacket, shirt, armor, outfit, watermark, text, dark image, low contrast"
)

# 确保是 **竖向全身图**
image = pipe(
    prompt,
    negative_prompt=negative_prompt,  # 防止黑屏
    num_inference_steps=50,  # 50步采样
    guidance_scale=7.5,  # 避免 Prompt 过度影响
    height=1024, width=1024  # **改为全身图尺寸**
).images[0]

def disable_safety_checker(images, clip_input):
    return images, [False] * len(images)

pipe.safety_checker = disable_safety_checker


# 保存 & 显示
image.save("fullbody_girl.png")
image.show()
