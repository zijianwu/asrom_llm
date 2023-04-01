from flask import Flask, make_response, request

from asrom_llm.qa import get_qa_v1
from asrom_llm.query_document import get_clinical_reference

app = Flask(__name__)


@app.route("/qa")
def search():
    query = request.args.get("query")  # Get the user's query from the POST request

    if not query:
        return make_response({"error": "No query provided"}, 400)

    # Call your main function that processes the query and returns the response
    response = get_qa_v1(query)

    return response, 200


@app.route("/clinical_reference")
def clinical_reference():
    query = request.args.get("query")

    if not query:
        return make_response({"error": "No query provided"}, 400)

    response = get_clinical_reference(query)

    return response, 200


if __name__ == "__main__":
    app.run(debug=False)
