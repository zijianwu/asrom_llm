import ast

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from asrom_llm.pubmed_search import parse_result, search_pubmed
from asrom_llm.utils import verbose_print

OUTLINE_SYSTEM_TEMPLATE = (
    "You are an expert medical writer for UpToDate. Given an article title, create an outline."
)
OUTLINE_EXAMPLES = [
    # TODO: Add more examples
    {
        "Title": "Superior Mesenteric Artery Syndrome",
        "Outline": '["SUMMARY AND RECOMMENDATIONS", "INTRODUCTION", "ANATOMY", "PATIENTS AT RISK", "CLINICAL EVALUATION", ["Differential diagnosis"], "DIAGNOSTIC STUDIES", ["Plain abdominal films", "Oral contrast studies", "Diagnostic imaging criteria"], "CONSERVATIVE THERAPY", ["Gastrointestinal decompression", "Correction of electrolyte abnormalities", "Nutrition support"], "SURGICAL MANAGEMENT", "OPEN SURGERY", ["Laparoscopic approach", "Follow-up", "Outcomes and complications of surgery"]]',
    },
    {
        "Title": "Management of Acute Pancreatitis",
        "Outline": '["SUMMARY AND RECOMMENDATIONS", "INTRODUCTION", "CLASSIFICATION", "ASSESSMENT OF DISEASE SEVERITY", ["Indications for monitored or intensive care"], "INITIAL MANAGEMENT", ["Fluid replacement", "Pain control", "Monitoring", "Nutrition", ["Oral", "Enteral", "Parenteral"], "Antibiotics", "Other therapies with no role"], "MANAGEMENT OF COMPLICATIONS", ["Local complications", ["Acute peripancreatic fluid collection", "Pancreatic pseudocyst", "Acute necrotic collection and walled-off necrosis"], "Peripancreatic vascular complications", ["Splanchnic venous thrombosis", "Pseudoaneurysm"], "Abdominal compartment syndrome", "Systemic complications"], "MANAGEMENT OF UNDERLYING PREDISPOSING CONDITIONS", ["Gallstone pancreatitis", ["Endoscopic retrograde cholangiopancreatography", "Cholecystectomy"], "Biliary sludge", "Hypertriglyceridemia", "Hypercalcemia", "Alcoholic pancreatitis"], "INFORMATION FOR PATIENTS"]',
    },
    {
        "Title": "Overview of Kidney Disease in Patients with Cancer",
        "Outline": '["SUMMARY", "INTRODUCTION", "EPIDEMIOLOGY AND PROGNOSIS", ["AKI in patients with cancer", "CKD in patients with cancer"], "ASSESSMENT OF KIDNEY FUNCTION IN PATIENTS WITH CANCER", "ACUTE KIDNEY INJURY IN PATIENTS WITH CANCER", ["Prerenal causes", "Intrinsic renal causes", ["Light chain cast nephropathy (formerly called myeloma kidney)", "Tumor lysis syndrome", "Tumor infiltration", "Thrombotic microangiopathy", "Nephrotoxic anticancer agents", "Less common etiologies"], "Postrenal causes", ["Intratubular obstruction", "Extrarenal obstruction"], "Special AKI populations", ["AKI after hematopoietic cell transplantation", "AKI after nephrectomy"]], "CHRONIC KIDNEY DISEASE IN PATIENTS WITH CANCER", ["Causes of CKD in patients with cancer", "Special CKD populations", ["Patients with renal cell carcinoma"]], "PROTEINURIA OR NEPHROTIC SYNDROME IN PATIENTS WITH CANCER", ["Paraneoplastic glomerular diseases", ["Membranous nephropathy", "Minimal change disease"], "Chemotherapy-associated glomerular disorders", "Disorders associated with monoclonal gammopathy", ["Amyloidosis", "Monoclonal immunoglobulin deposition disease", "Less common causes"]], "ELECTROLYTE DISORDERS IN PATIENTS WITH CANCER", ["Hyponatremia", "Hypernatremia", "Hypercalcemia", "Hypokalemia", "Hyperkalemia", "Hypophosphatemia", "Hyperphosphatemia", "Hypomagnesemia"]]',
    },
]
OUTLINE_EXAMPLE_TEMPLATE = """
Title: {Title}
========
Outline: {Outline}
"""

SECTION_SYSTEM_TEMPLATE = "You are a medical textbook author. Given the following sources and a section header, write a concise section. DO NOT INCLUDE ANY INFORMATION THAT DOES NOT DIRECTLY RELATE TO THE SECTION HEADER. ONLY INCLUDE INFORMATION RELATED TO MOST SPECIFIC PORTION OF THE SECTION HEADER. Do not include information that is not found within the sources."
SECTION_EXAMPLES = [
    # TODO: Add more examples
    {
        "section_header": "Superior Mesenteric Artery Syndrome - Clinical Evaluation",
        "sources": "[('4008904', 'Intestinal obstruction of the duodenum by entrapment between the aorta and the superior mesenteric artery (SMA) is an uncommon cause of megaduodenum. Despite many case reports, acceptance of the SMA syndrome as a clinical entity has been controversial on account of its confusion with other causes of megaduodenum. We therefore report a case of SMA syndrome which sharply exemplifies its clinical and anatomic features. The clinical findings are proximal duodenal obstruction with an abrupt cutoff and active peristalsis. The anatomic features of this entity are a narrow angle between the aorta and the SMA, together with high fixation of the duodenum by the ligament of Treitz and/or an anomalous SMA crossing directly over the aorta at its intersection with the duodenum. The SMA syndrome may occur as an acute self-limited event due to a reversible precipitating factor, or as a chronic recurring disorder. The acute form subsides with correction of the specific initiating factor; the chronic form responds favorably to simple surgical mobilization of the duodenum.'), ('17062207', 'A 50-year-old man who underwent esophagectomy with cervical esophagogastrostomy for esophageal cancer presented with superior mesenteric artery syndrome. He had severe diffuse and dull abdominal pain and frequent vomiting that began within 10 days after the operation. He also complained of indigestion and early fullness, and lost more than 5 kg of body weight during the period. The symptoms were initiated by poor oral intake and weight loss, and were relieved by nutritional support. Although it is very rare, we conclude that surgeons and radiologists should be aware of the possibility of superior mesenteric artery syndrome as one of the complications after esophagectomy.'), ('11518108', 'We report two identical male twins who suffered from superior mesenteric artery (SMA) syndrome. A 28-year-old man was admitted for investigation of postprandial nausea and vomiting. Upper gastrointestinal examination revealed a dilated proximal duodenum with an abrupt vertical cutoff of barium flow in the third portion of the duodenum, establishing the diagnosis of SMA syndrome. One year later, his twin brother also presented similar symptoms and was radiologically diagnosed as SMA syndrome. The twin brothers did not respond adequately to conservative therapy and underwent duodenojejunostomy. This is the first report of SMA syndrome in identical twins.')]",
        "section_with_references": "Patients may present acutely (such as following surgery) or more insidiously with progressive symptoms [4008904]. In both cases, symptoms are consistent with proximal small bowel obstruction. Patients with mild obstruction may have only postprandial epigastric pain and early satiety, while those with more advanced obstruction may have severe nausea, bilious emesis and weight loss. Patients may also have symptoms of reflux. \n\nFindings on physical examination are nonspecific but can include abdominal distension, a succussion splash, and high-pitched bowel sounds. Laboratory examination can be normal or, in patients with severe vomiting, significant electrolyte abnormalities may be present. \n\n Diagnosis is often delayed and may result in significant complications including [17062207, 11518108]: \n●Fatalities due to electrolyte abnormalities \n●Fatalities due to gastric perforation \n●Gastric pneumatosis and portal venous gas ●Formation of an obstructing duodenal bezoar",
    },
]
SECTION_EXAMPLE_TEMPLATE = """
Section Header: {section_header}
========
Sources: {sources}
========
Section with References: {section_with_references}
"""


class Section:
    """
    Represents a section in a document outline with title, content, and children sections.
    """

    def __init__(self, title, content=None, prev_titles=None):
        self.title = title
        self.content = content
        if prev_titles:
            self.full_title = f"{prev_titles} - {title}"
        else:
            self.full_title = title
        self.children = []

    def add_child(self, child_section):
        """
        Adds a child section to the current section.

        Args:
            child_section (Section): The child section to be added.
        """
        self.children.append(child_section)

    def __str__(self):
        return f"{self.title} (has content {self.content is not None})"


class DocumentOutline:
    """
    Represents a clinical reference about a given a title, generated with the
    provided model name, temperature, and maximum tokens.
    """

    def __init__(
        self,
        title,
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=256,
    ):
        """
        Initializes a DocumentOutline object with the given title, model name, temperature, and maximum tokens.

        Args:
            title (str): The title of the document outline.
            model_name (str, optional): The name of the model to use. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): The temperature to use for content generation. Defaults to 0.3.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.
        """
        self.root = Section(title)
        self.title = title
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # TODO: Add ability to change models
        self.llm = ChatOpenAI(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
        self._create_toc(title)

    def add_heading(self, heading_title, content, parent_title=None):
        """
        Adds a heading to the document outline, either at the root level or as a child of another section.

        Args:
            heading_title (str): The title of the heading to add.
            content (str): The content of the heading.
            parent_title (str, optional): The title of the parent section, if any. Defaults to None.
        """
        if parent_title is None:
            self.root.add_child(Section(heading_title, content, prev_titles=self.root.full_title))
        else:
            parent_section = self.find_section_by_title(parent_title, self.root)
            parent_section.add_child(
                Section(heading_title, content, prev_titles=parent_section.full_title)
            )

    def find_section_by_title(self, title, section=None):
        """
        Finds a section by its title in the document outline, starting from the given section.

        Args:
            title (str): The title of the section to find.
            section (Section, optional): The starting section for the search. Defaults to None.

        Returns:
            Section: The section with the given title, or None if not found.
        """
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
        """
        Prints the document outline
        """
        self.print_section(self.root)

    def print_section(self, section, depth=0):
        """
        Prints a section in the document outline.
        """
        print("\t" * depth + str(section))
        for child in section.children:
            self.print_section(child, depth + 1)

    def _create_toc_from_outline(self, outline, parent_title=None):
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
                    self._create_toc_from_outline(item, parent_title=outline[i - 1])
        return self

    def to_markdown(self, section=None, depth=1):
        """
        Output the entire clinical reference document (outline and content)
        to a Markdown formatted string.

        Args:
        - section (Section, optional): The current section to process. Defaults to None.
        - depth (int, optional): The depth of the current section in the outline. Defaults to 1.

        Returns:
        - str: The Markdown formatted string representation of the document outline.
        """
        if section is None:
            section = self.root
            md_str = f"# {self.title.title()}\n\n"
        else:
            md_str = ""
        if section != self.root:
            md_str += f"{'#' * depth} {section.title.title()}\n"
            if section.content:
                md_str += f"{section.content}\n\n"

        for child in section.children:
            md_str += self.to_markdown(child, depth + 1)

        return md_str

    def _get_section(self, section_header_input, sources_input):
        """
        Generate content for a section using the given section header and sources.

        Args:
        - section_header_input (str): The header for the section.
        - sources_input (str): A string containing the sources for the section.

        Returns:
        - str: The generated content for the section.
        """
        section_example_prompt = PromptTemplate(
            input_variables=[
                "section_header",
                "sources",
                "section_with_references",
            ],
            template=SECTION_EXAMPLE_TEMPLATE,
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(SECTION_SYSTEM_TEMPLATE)
        section_human_prompt = FewShotPromptTemplate(
            # TODO: Add a dynamic example selector to allow for longer inputs/responses
            examples=SECTION_EXAMPLES,
            example_prompt=section_example_prompt,
            suffix="Section Header: {section_header_input}\n========\nSources: {sources_input}\n========\nSection with References: ",
            input_variables=["section_header_input", "sources_input"],
            example_separator="\n\n",
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=section_human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        outline_chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        output = outline_chain.run(
            section_header_input=section_header_input,
            sources_input=sources_input,
        )
        return output

    def _get_pubmed_sources(self, query):
        """
        Search PubMed for sources based on the given query.

        Args:
        - query (str): The query to search PubMed for.

        Returns:
        - list: A list of tuples containing PMID and abstracts of the search results.
        """
        # TODO: Increase page_size after dynamic example selector created
        # TODO: Improve query prior to searching Pubmed API
        # TODO: Add ability to pull in full text articles through PMC
        records = search_pubmed(query, page_num=1, page_size=3)
        parsed_result = parse_result(records)
        if len(parsed_result) < 2:
            query = query.split(" - ")[0] + " - " + query.split(" - ")[-1]
            records = search_pubmed(query, page_num=1, page_size=4)
            parsed_result = parse_result(records)
        result = parsed_result.set_index("PMID")["Abstract"].to_dict()
        result = result.items()
        return result

    def add_content(self, section=None, verbose=False):
        """
        Add content to the sections of the document outline.
        Default is to add content to all sections of the document.

        Args:
        - section (Section, optional): The current section to process. Defaults to None.
        - verbose (bool, optional): If True, print the full title of each section. Defaults to False.

        Returns:
        - DocumentOutline: The updated DocumentOutline instance with content added.
        """
        # TODO: Add to vector database search
        # TODO: Create async calls so sections can be generated faster rather than sequentially
        if section is None:
            section = self.root
        if not section.children:
            verbose_print(section.full_title, verbose)
            section.content = self._get_section(
                section.full_title,
                str(self._get_pubmed_sources(section.full_title))
                .replace("{", "\{")
                .replace("}", "\}"),
            )
        else:
            for child in section.children:
                self.add_content(child)
        return self

    def _get_outline(self, title):
        """
        Generate an outline for the document based on the given title.

        Args:
        - title (str): The title of the document.

        Returns:
        - list: A nested list representing the generated outline structure.
        """
        outline_example_prompt = PromptTemplate(
            input_variables=["Title", "Outline"], template=OUTLINE_EXAMPLE_TEMPLATE
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(OUTLINE_SYSTEM_TEMPLATE)
        outline_human_prompt = FewShotPromptTemplate(
            examples=OUTLINE_EXAMPLES,
            example_prompt=outline_example_prompt,
            suffix="Title: {input}\n========\nOutline: ",
            input_variables=["input"],
            example_separator="\n\n",
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=outline_human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        outline_chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        output = outline_chain.run(input=title)
        try:
            result = ast.literal_eval(output)
        except SyntaxError as err:
            # TODO: Implement OutputFixingParser
            print("Need to implement version of OutputFixingParser", err)
        return result

    def _create_toc(self, title):
        """
        Initialize the table of contents for the document using the given title.

        Args:
        - title (str): The title of the document.

        Returns:
        - None
        """
        outline = self._get_outline(title)
        self._create_toc_from_outline(outline)
