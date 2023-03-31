from asrom_llm.document import DocumentOutline, Section


def test_section_init():
    section = Section("Test Title", content="Test Content")
    assert section.title == "Test Title"
    assert section.content == "Test Content"
    assert len(section.children) == 0


def test_section_add_child():
    parent_section = Section("Parent Title")
    child_section = Section("Child Title")
    parent_section.add_child(child_section)

    assert len(parent_section.children) == 1
    assert parent_section.children[0] == child_section


def test_section_full_title():
    # Test without prev_titles
    section1 = Section("Introduction", content="Introduction content")
    assert section1.full_title == "Introduction"

    # Test with prev_titles
    prev_titles = "Chapter 1"
    section2 = Section(
        "Section 1.1", content="Section 1.1 content", prev_titles=prev_titles
    )
    assert section2.full_title == f"{prev_titles} - Section 1.1"

    # Test with updated prev_titles
    prev_titles = "Chapter 1 - Section 1.1"
    section3 = Section(
        "Subsection 1.1.1",
        content="Subsection 1.1.1 content",
        prev_titles=prev_titles,
    )
    assert section3.full_title == f"{prev_titles} - Subsection 1.1.1"


def test_section_str_representation():
    section_with_content = Section(
        "Title with content", content="Some content"
    )
    section_without_content = Section("Title without content")

    assert str(section_with_content) == "Title with content (Has content True)"
    assert (
        str(section_without_content)
        == "Title without content (Has content False)"
    )


def test_document_outline_init():
    doc_outline = DocumentOutline("Test TOC")
    assert doc_outline.root.title == "Test TOC"


def test_document_outline_add_heading():
    doc_outline = DocumentOutline("Test TOC")
    doc_outline.add_heading("Heading 1", None)
    assert len(doc_outline.root.children) == 1
    assert doc_outline.root.children[0].title == "Heading 1"


def test_document_outline_find_section_by_title():
    doc_outline = DocumentOutline("Test TOC")
    doc_outline.add_heading("Heading 1", None)
    doc_outline.add_heading("Subheading 1", None, parent_title="Heading 1")

    heading_1 = doc_outline.find_section_by_title(
        "Heading 1", doc_outline.root
    )
    subheading_1 = doc_outline.find_section_by_title(
        "Subheading 1", doc_outline.root
    )

    assert heading_1.title == "Heading 1"
    assert subheading_1.title == "Subheading 1"


def test_document_outline_create_doc_outline_from_outline():
    outline = ["Heading 1", ["Subheading 1"], "Heading 2", ["Subheading 2"]]
    doc_outline = DocumentOutline("Test TOC")
    doc_outline.create_toc_from_outline(outline)

    heading_1 = doc_outline.find_section_by_title(
        "Heading 1", doc_outline.root
    )
    subheading_1 = doc_outline.find_section_by_title(
        "Subheading 1", doc_outline.root
    )
    heading_2 = doc_outline.find_section_by_title(
        "Heading 2", doc_outline.root
    )
    subheading_2 = doc_outline.find_section_by_title(
        "Subheading 2", doc_outline.root
    )

    assert heading_1.title == "Heading 1"
    assert subheading_1.title == "Subheading 1"
    assert heading_2.title == "Heading 2"
    assert subheading_2.title == "Subheading 2"
