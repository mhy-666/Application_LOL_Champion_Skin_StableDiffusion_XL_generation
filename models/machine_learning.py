import nltk
from sentence_transformers import SentenceTransformer
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextImageGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the TextImageGenerator class.
        
        Args:
        model_name (str): The name of the pre-trained sentence encoding model.
        """
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('brown')
        
        # Load the pre-trained sentence encoding model
        self.model = SentenceTransformer(model_name)
        
        # Load n-gram model
        self.n_gram_model = nltk.lm.Laplace(3)
        self.n_gram_model.train(nltk.corpus.brown.sents())

    def generate_text(self, start_text, num_words=10):
        """
        Generate text using n-gram language model.
        
        Args:
        start_text (str): The starting text for generation.
        num_words (int): The number of words to generate.
        
        Returns:
        str: The generated text.
        """
        tokens = nltk.word_tokenize(start_text)
        for _ in range(num_words):
            next_token = self.n_gram_model.generate(tokens[-2:])
            tokens.append(next_token)
        generated_text = ' '.join(tokens)
        return generated_text

    def find_similar_images(self, text_description, image_dir, tag_file):
        """
        Find the most similar images to a given text description in a specified image directory.
        
        Args:
        text_description (str): The text description.
        image_dir (str): The path to the image directory.
        tag_file (str): The path to the file containing image tags.
        
        Returns:
        list: A list of paths to the most similar images.
        """
        # Encode the text description
        text_encoding = self.model.encode([text_description])
        
        # Read the image tags
        with open(tag_file, 'r') as f:
            data = [json.loads(line) for line in f]
        tags = [entry['prompt'] for entry in data]
        
        # Encode the image tags
        tag_encodings = self.model.encode(tags)
        
        # Calculate the cosine similarity between the text description and each image tag
        similarities = cosine_similarity(text_encoding, tag_encodings).flatten()
        
        # Get the indices of the top 5 most similar images
        top_indices = similarities.argsort()[-5:][::-1]
        
        # Return the paths to the most similar images
        similar_images = [os.path.join(image_dir, data[idx]['file_name']) for idx in top_indices]
        
        return similar_images

def main():

    start_text = "In league of legend, a cute little teemo"
    num_words = 5
    image_dir = "./data/raw/"
    tag_file = "/data/processed/tag/file.jsonl"

    # Create an instance of the TextImageGenerator class
    text_image_generator = TextImageGenerator()

    # Generate text
    generated_text = text_image_generator.generate_text(start_text, num_words)
    print("Generated text:", generated_text)

    # Find the most similar images
    similar_images = text_image_generator.find_similar_images(generated_text, image_dir, tag_file)
    print("Most similar images:")
    for img_path in similar_images:
        print(img_path)

if __name__ == "__main__":
    main()