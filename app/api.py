from flask import Flask, make_response, request

from asrom_llm.qa import get_qa_v1

app = Flask(__name__)


@app.route("/search")
def search():
    query = request.args.get(
        "query"
    )  # Get the user's query from the POST request

    if not query:
        return make_response({"error": "No query provided"}, 400)

    # Call your main function that processes the query and returns the response
    response = get_qa_v1(query)

    return response, 200


if __name__ == "__main__":
    app.run(debug=False)
