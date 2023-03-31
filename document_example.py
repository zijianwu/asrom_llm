from asrom_llm.document import DocumentOutline
from asrom_llm.query_document import get_outline

TITLE = "Chronic mesenteric ischemia"
outline = get_outline(title=TITLE)
document = DocumentOutline(title=TITLE).create_toc_from_outline(outline)
document.print_toc()
