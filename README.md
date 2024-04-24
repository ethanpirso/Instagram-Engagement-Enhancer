# Instagram Engagement Enhancer
This project leverages the Google Vision API and Latent Dirichlet Allocation (LDA) for topic modeling to analyze Instagram images. By evaluating the association between image content and user engagement, we provide data-driven recommendations to help influencers increase interaction on their posts.

## Setup
1. Install required packages:
   ```bash
   pip install google-cloud-vision gensim
   ```
   
2. Set up Google Cloud Vision API:
Follow the instructions at Google Cloud Documentation to set up your API key.

## Running the Analysis
To run the analysis, execute the main.py script:
   ```bash
    python main.py
   ```
 
## Modules
- google_vision.py: Interfaces with Google Vision API to label images.
- topic_modeling.py: Performs LDA to generate and analyze topics from image labels.
- engagement_analysis.py: Compares topic distributions across different levels of engagement.
- main.py: Orchestrates the analysis workflow.

This setup provides a comprehensive guide to managing your analysis workflow. Adjust the paths, parameters, and specific functions to align with your actual data and requirements.
