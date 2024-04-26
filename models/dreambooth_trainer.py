import os
import locale
import json
from pathlib import Path
from slugify import slugify
from huggingface_hub import notebook_login, hf_hub_download, upload_file, whoami
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
import glob
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

class DreamboothTrainer:
   def __init__(self):
       self.model_name = "LOL champion skin SDXL LoRA4"
       self.output_dir = './data/outputs'
       self.instance_prompt = "champion in the style of leagueoflegend"
       self.validation_prompt = "ahri is walking on the beach, in the style of leagueoflegend"
       self.rank = 8

   def prepare_dataset(self, local_dir, dataset_to_download=None, caption_prefix="champion in the style of leagueoflegend, "):
       if dataset_to_download:
           os.makedirs(local_dir, exist_ok=True)
           from huggingface_hub import snapshot_download
           snapshot_download(
               dataset_to_download,
               local_dir=local_dir,
               repo_type="dataset",
               ignore_patterns=".gitattributes",
           )

       

       device = "cuda" if torch.cuda.is_available() else "cpu"
       blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
       blip_model = Blip2ForConditionalGeneration.from_pretrained(
           "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
       ).to(device)

       def caption_images(input_image):
           inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
           pixel_values = inputs.pixel_values

           generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
           generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
           return generated_caption

       imgs_and_paths = [(path, Image.open(path)) for path in glob.glob(f"{local_dir}*.jpg")]

       with open(f"{local_dir}metadata.jsonl", "w") as outfile:
           for img in imgs_and_paths:
               caption = caption_prefix + img[0].split("/")[-1].split("_")[0] + ", " + caption_images(img[1]).split("/m")[0]
               entry = {"file_name": img[0].split("/")[-1], "prompt": caption}
               json.dump(entry, outfile)
               outfile.write("\n")

       torch.cuda.empty_cache()

   def train(self):
       locale.getpreferredencoding = lambda: "UTF-8"
       os.system("accelerate config default")


       os.system(
           f"accelerate launch train_dreambooth_lora_sdxl_advanced.py "
           f"--pretrained_model_name_or_path='stabilityai/stable-diffusion-xl-base-1.0' "
           f"--pretrained_vae_model_name_or_path='madebyollin/sdxl-vae-fp16-fix' "
           f"--dataset_name='/content/drive/MyDrive/lol_champion_skin/more_data' "
           f"--instance_prompt='{self.instance_prompt}' "
           f"--validation_prompt='{self.validation_prompt}' "
           f"--output_dir='{self.output_dir}' "
           f"--caption_column='prompt' "
           f"--mixed_precision='fp16' "
           f"--resolution=512 "
           f"--train_batch_size=3 "
           f"--repeats=1 "
           f"--report_to='wandb' "
           f"--gradient_accumulation_steps=1 "
           f"--gradient_checkpointing "
           f"--learning_rate=1.0 "
           f"--text_encoder_lr=1.0 "
           f"--adam_beta2=0.99 "
           f"--optimizer='prodigy' "
           f"--train_text_encoder_ti "
           f"--train_text_encoder_ti_frac=0.5 "
           f"--snr_gamma=5.0 "
           f"--lr_scheduler='constant' "
           f"--lr_warmup_steps=0 "
           f"--rank={self.rank} "
           f"--max_train_steps=16000 "
           f"--checkpointing_steps=200 "
           f"--seed='0' "
           f"--push_to_hub "
           f"--resume_from_checkpoint='./checkpoint-12000'"
       )

   def inference(self):
       username = whoami(token=Path("/root/.cache/huggingface/"))["name"]
       repo_id = f"{username}/{self.output_dir}"

       pipe = DiffusionPipeline.from_pretrained(
           "stabilityai/stable-diffusion-xl-base-1.0",
           torch_dtype=torch.float16,
           variant="fp16",
       ).to("cuda")

       pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")

       text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
       tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

       embedding_path = hf_hub_download(repo_id=repo_id, filename="embeddings.safetensors", repo_type="model")

       state_dict = load_file(embedding_path)
       pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
       pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

       instance_token = "<s0><s1>"
       prompt = f"a {instance_token} icon of an orange llama eating ramen, in the style of {instance_token}"

       image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
       image.show()

def main():
   trainer = DreamboothTrainer()
   local_dir = "./data/training_data"
   dataset_to_download = "wintercoming6/lol_champion_skin"
   trainer.prepare_dataset(local_dir, dataset_to_download)
   trainer.train()
   trainer.inference()

if __name__ == "__main__":
   main()