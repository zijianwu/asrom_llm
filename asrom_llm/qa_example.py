from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI

from asrom_llm.pubmed_search import parse_result, search_pubmed

QUERY = "What are some non‐pharmacological interventions for headache in children and adolescents?"

llm = OpenAI(
    model_name="gpt-3.5-turbo",
    organization="org-ANpnkjrEDLjbQriFNhWllHVQ",
    temperature=0.5,
)

records = search_pubmed(QUERY, page_num=1, page_size=2)
parsed_result = parse_result(records)
docs = [
    Document(page_content=record["Abstract"])
    for _, record in parsed_result.iterrows()
]

chain = load_qa_chain(llm, chain_type="stuff")
print(chain.run(input_documents=docs, question=QUERY))
