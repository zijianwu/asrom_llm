# ASROM LLM

ASROM LLM  provides answers to medical questions by searching through articles in Pubmed. This repository allows users to quickly and efficiently search through Pubmed articles to find answers to their medical questions.

## Installation
To install ASROM LLM, you will need to have poetry installed on your machine. Follow these steps to install poetry:
1. Install poetry using pip:
    
    ```pip install poetry```
1. Clone the repository from Github
    
    ```git clone git@github.com:zijianwu/asrom_llm.git```
1. Install the required packages using poetry:
    ```
    cd asrom_llm
    poetry install
    ```
1. Set your OpenAI key as a global variable:
    
    ```export OPENAI_API_KEY='...'```

## Testing the web interface
No data is saved during use of the web interface; it used purely for demo purposes. Follow these steps to run the website:
1. Ensure that you are in the correct environment where ASROM LLM is installed
    
    ```poetry shell```
1. Run the ASROM LLM website:
    
    ```python app/api.py & streamlit run app/app.py```
1. Go to the ASROM LLM website at http://localhost:8501 with the following test account information
    ```
    username: user
    password: password
    ```

## Experimentation
In order to experiment with different configurations, you can add additional queries to the `qa_example.py` file. You can then run `python qa_example.py` and look at the results that are saved in `results.json`

### Creating and running the Milvus vector database
Some of the QA methods require searching Pubmed indexed and in a Milvus database. Follow the steps below to create and run the Milvus database:
1. Ensure that you are in the correct environment where ASROM LLM is installed
    
    ```poetry shell```
1. Start the local database
    
    ```sudo docker-compose up -d```
1. If the database is NOT constructed, create and index the Pubmed articles
   
   ```python construct_database.py```
