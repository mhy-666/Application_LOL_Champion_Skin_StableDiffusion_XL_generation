# Application_LOL_Champion_Skin_StableDiffusion_XL_generation

The League of Legends (LoL) universe, developed by Riot Games, is a content-rich and expansive world filled with unique champions and their skins (visual representations). Although the existing universe is already broad, the potential for expansion is limitless. This project aims to create new, AI-generated content to further enrich the LoL universe, including new champion skins, background stories, and corresponding visual representations of these stories. Given the recent release of OpenAI's latest product, Sora, I am considering generating animations for the corresponding champions and background stories based on the content created, but since OpenAI has not yet released an API for Sora, this additional content is not guaranteed to be completed, but I will make every effort to do so.

## Previous Efforts

There have been multiple attempts to use artificial intelligence to expand game universes. Notably, OpenAI's GPT models have been used to generate textual content such as stories and dialogues. Similarly, image generation models like DALL-E have been utilized to create visual content based on textual prompts. Likewise, various models fine-tuned using the Stable Diffusion model can be found on platforms like Hugging Face (https://huggingface.co/models) and Civitai (https://civitai.com/models/). However, AI-generated content exploration in a gaming context is less common, offering a vast field for creative expansion in detail-rich universes like LoL, which boasts as many as 167 champions, providing a large space for expansion.

## Data Sources

The primary data sources for this project will include:

- Publicly available datasets of LoL champion images and skin artworks. (https://www.skinexplorer.lol/)
- Textual descriptions of champions and their lore from the official LoL universe page. (https://universe.leagueoflegends.com/en_US/story/)
- Official LoL fan-made comics (as an alternative method in case of insufficient data). (https://universe.leagueoflegends.com/en_US/comic/)
- Fan-made textual works on fan websites. (https://www.reddit.com/r/leagueoflegends/comments/loj1xo/collected_the_lore_on_universe_into_an_ebook/)

## Proposed Approach

Our approach diverges from previous efforts by combining both textual and visual AI models to achieve an expansion of the LoL universe. Specifically, we plan to:

1. **Dataset Creation**: Compile a dataset of LoL champions and their skins, including images and a dataset of textual lore for RAG database.
2. **Textual Content Generation**: Use state-of-the-art language models (e.g., GPT-4 or similar LLMs) to generate new background stories for champions.
3. **RAG Database Creation**: Utilize Retrieval-Augmented Generation (RAG) technology to establish a database storing the background stories of LoL champions, aiding large language models in better understanding the context and generating corresponding text.
4. **Visual Content Generation**: Use image-to-text model, like the BLIP2 model (https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/blip-2) to tag existing images with relevant prompts and employ the Stable Diffusion model (potentially with fine-tuning) to generate new champion skins and visual representations of the AI-generated stories.

## Unique Contribution

The unique contribution of this project lies in the integration of textual and visual AI models to create a multi-modal expansion of the LoL universe. By generating new lore and visual representations of that lore, we aim to showcase a new method for expanding game universes that could be applied to other game series.
