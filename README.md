# LoL Champion Skin StableDiffusion XL Generation

## Problem Statement

The League of Legends (LoL) universe, created by Riot Games, is a rich and expansive world filled with unique champions and their skins (visual representations). While the existing universe is already vast, there is an opportunity to further enrich it by generating new champion skins, background stories, and corresponding visual representations of these stories. This project aims to leverage artificial intelligence to create new content for the LoL universe, showcasing a novel method for expanding game universes that could be applied to other game series.

## Data Sources

The primary data sources for this project include:

- Publicly available datasets of LoL champion images and skin artworks from [SkinExplorer](https://www.skinexplorer.lol/).
- Textual descriptions of champions and their lore from the official LoL universe page: [Universe of League of Legends](https://universe.leagueoflegends.com/en_US/story/).
- Official LoL fan-made comics as an alternative data source: [Universe of League of Legends Comics](https://universe.leagueoflegends.com/en_US/comic/).
- Fan-made textual works from websites like [Reddit](https://www.reddit.com/r/leagueoflegends/comments/loj1xo/collected_the_lore_on_universe_into_an_ebook/).

## Previous Efforts and Literature Review

Previous efforts have been made to use artificial intelligence to expand game universes. Notable examples include:

- OpenAI's GPT models have been used to generate textual content such as stories and dialogues.
- Image generation models like DALL-E have been utilized to create visual content based on textual prompts.
- Various models fine-tuned using the Stable Diffusion model can be found on platforms like Hugging Face (https://huggingface.co/models) and Civitai (https://civitai.com/models).

However, AI-generated content exploration in a gaming context is less common, leaving a vast field for creative expansion in detail-rich universes like LoL, which boasts as many as 167 champions, providing a large space for expansion.

## Model Evaluation Process & Metric Selection

For the text generation component, the following evaluation process was employed:

- Comparison between ChatGPT models with and without a Retrieval-Augmented Generation (RAG) database.
- Evaluation of the generated stories using AI assistance from [AI Evaluate AI](https://ai-evaluate-ai.cloudcv.org/) to assess the quality and creativity of the generated stories.

For the visual content generation component:

- Comparison between fine-tuned and non-fine-tuned Stable Diffusion XL models.
- Human evaluation by three LoL game players, including myself, to assess the plausibility and relevance of the generated champion skin images.

## Modeling Approach

The project combines both textual and visual AI models to achieve an expansion of the LoL universe:

1. **Dataset Creation**: Compile a dataset of LoL champions and their skins, including images and a dataset of textual lore for the RAG database.
2. **Textual Content Generation**: Use state-of-the-art language models (e.g., GPT-4 or similar LLMs) to generate new background stories for champions.
3. **RAG Database Creation**: Utilize Retrieval-Augmented Generation (RAG) technology to establish a database storing the background stories of LoL champions, aiding large language models in better understanding the context and generating corresponding text.
4. **Visual Content Generation**: Use an image-to-text model like BLIP2 (https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/blip-2) to tag existing images with relevant prompts, and employ the Stable Diffusion XL model (potentially with fine-tuning) to generate new champion skins and visual representations of the AI-generated stories.

## Data Processing Pipeline

1. **Text Data Processing**: Clean and preprocess the textual data from various sources, including official lore descriptions and fan-made works. This step may involve techniques such as text normalization, tokenization, and filtering.
2. **Image Data Processing**: Preprocess the image data by resizing, converting to a compatible format, and applying any necessary transformations.
3. **RAG Database Creation**: Construct the RAG database by indexing the preprocessed textual data, allowing for efficient retrieval during text generation.

## Models Evaluated

### Non-Deep Learning Model
- **N-gram Language Model**: For the text generation component, an n-gram language model was used as a traditional machine learning approach. The n-gram model's output was used as a prompt for image generation.

### Deep Learning Models
- **Prompt Encoding and Cosine Similarity**: For the image generation component, existing images were tagged using prompt encoding with the BLIP2 model. The cosine similarity between the encoded prompts and image tags was used to search for the most relevant images. However, this approach has the limitation of being unable to generate new images.

- **Stable Diffusion XL**: The Stable Diffusion XL model was employed for generating new champion skins and visual representations of the AI-generated stories. Fine-tuning was explored to improve the model's performance on the LoL domain.

## Comparison to Naive Approach

As a naive approach, a set of pre-existing LoL comics was stored, and a random comic was selected and displayed for each request. This approach serves as a baseline for comparison with the AI-generated content.

## Demo

You can access demo [here](https://applicationlolchampionskinstablediffusionxlgeneration-knhkov9c.streamlit.app/)

## Results and Conclusions

[Provide a summary of your results, key findings, and conclusions based on the model evaluations and comparisons.]

## How to Run

To run this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-repo.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Prepare the data:
   - Download the LoL champion images and skin artworks from [SkinExplorer](https://www.skinexplorer.lol/).
   - Collect the textual descriptions of champions and their lore from the official LoL universe page: [Universe of League of Legends](https://universe.leagueoflegends.com/en_US/story/).
   - (Optional) Gather additional data from other sources, such as fan-made comics and textual works.
4. Preprocess the data by running the following command: `python preprocess.py`
5. Train the models:
   - For text generation: `python train_text_model.py`
   - For image generation: `python train_image_model.py`
6. Generate new content:
   - For text generation: `python generate_text.py`
   - For image generation: `python generate_images.py`

Note: The actual file names and commands may vary based on your implementation. Please refer to the project documentation for more detailed instructions.
