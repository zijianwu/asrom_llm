from asrom_llm.document import DocumentOutline

TITLE = "Chronic mesenteric ischemia"
document = DocumentOutline(
    title=TITLE, model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=256
)
document = document.add_content(verbose=True)
