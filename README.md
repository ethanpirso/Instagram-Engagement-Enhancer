# Instagram Engagement Enhancer
This project leverages the Google Vision API and Latent Dirichlet Allocation (LDA) for topic modeling to analyze Instagram images. By evaluating the association between image content and user engagement, we provide data-driven recommendations to help influencers increase interaction on their posts.

## Setup
1. Install required packages:
   ```bash
   pip install google-cloud-vision gensim pandas openai matplotlib seaborn
   ```
   
2. Set up Google Cloud Vision API:
Follow the instructions at Google Cloud Documentation to set up your API key.

3. Set up OpenAI API:
   - Obtain an API key from OpenAI by creating an account and following the instructions on the OpenAI website.
   - Set up your API key in your environment:
     ```bash
     export OPENAI_API_KEY='your_api_key_here'
     ```

## Running the Analysis
To run the analysis, execute the main.ipynb notebook.
 
## Modules
- google_vision.py: Interfaces with Google Vision API to label images.
- topic_modeling.py: Performs LDA to generate and analyze topics from image labels.
- main.ipynb: Orchestrates the analysis workflow.
