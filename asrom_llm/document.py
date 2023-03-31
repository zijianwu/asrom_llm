class Section:
    def __init__(self, title, content=None, prev_titles=None):
        self.title = title
        self.content = content
        if prev_titles:
            self.full_title = f"{prev_titles} - {title}"
        else:
            self.full_title = title
        self.children = []

    def add_child(self, child_section):
        self.children.append(child_section)

    def __str__(self):
        return f"{self.title} (has content {self.content is not None})"


class DocumentOutline:
    def __init__(self, title):
        self.root = Section(title)

    def add_heading(self, heading_title, content, parent_title=None):
        if parent_title is None:
            self.root.add_child(
                Section(
                    heading_title, content, prev_titles=self.root.full_title
                )
            )
        else:
            parent_section = self.find_section_by_title(
                parent_title, self.root
            )
            parent_section.add_child(
                Section(
                    heading_title,
                    content,
                    prev_titles=parent_section.full_title,
                )
            )

    def find_section_by_title(self, title, section=None):
        if section is None:
            section = self.root
        if section.title == title:
            return section
        for child in section.children:
            result = self.find_section_by_title(title, child)
            if result is not None:
                return result
        return None

    def print_toc(self):
        self.print_section(self.root)

    def print_section(self, section, depth=0):
        print("\t" * depth + str(section))
        for child in section.children:
            self.print_section(child, depth + 1)

    def create_toc_from_outline(self, outline, parent_title=None):
        """
        Create a DocumentOutline from an outline in the form of a list of lists.

        Takes the outline (a nested list) and creates a DocumentOutline instance
        with the outline structure. Each element in the outline is added as a
        heading to the DocumentOutline, with optional parent title for nested
        headings.

        Args:
        - outline (list): A nested list representing the outline structure.
        - parent_title (str, optional): The title of the parent section.

        Returns:
        - DocumentOutline: An updated DocumentOutline instance.
        """
        if isinstance(outline, str):
            self.add_heading(outline, None, parent_title)
        else:
            for i, item in enumerate(outline):
                if isinstance(item, str):
                    self.add_heading(item, None, parent_title)
                else:
                    self.create_toc_from_outline(
                        item, parent_title=outline[i - 1]
                    )
        return self
