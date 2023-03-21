Directions:
1. Clone the repo
2. Install Poetry (https://python-poetry.org/docs/)
3. Run `poetry install` within the repo
4. Set your OpenAI key as a global variable by running `export OPENAI_API_KEY='...'`
5. Run `qa_example.py`; you can change the QUERY to one of your choice in here

In order to create the Milvus vector database locally, adding in the PubMed Medline Baseline files:
1. Run `sudo docker-compose up -d` to start the database
2. Run `construct_database.py`
