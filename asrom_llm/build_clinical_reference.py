from asrom_llm.clinical_reference import DocumentOutline


def get_clinical_reference(query):
    # TODO: Increase response max_tokens after dynamic example selection created
    document = DocumentOutline(
        title=query, model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=256
    )
    document = document.add_content(verbose=True)
    markdown = document.to_markdown()
    return markdown
