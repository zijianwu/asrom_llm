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

OUTLINE_SYSTEM_TEMPLATE = "You are an expert medical writer for UpToDate. Given an article title, create an outline."
OUTLINE_EXAMPLES = [
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


def get_outline(title="Chronic mesenteric ischemia"):
    outline_example_prompt = PromptTemplate(
        input_variables=["Title", "Outline"], template=OUTLINE_EXAMPLE_TEMPLATE
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        OUTLINE_SYSTEM_TEMPLATE
    )
    outline_human_prompt = FewShotPromptTemplate(
        examples=OUTLINE_EXAMPLES,
        example_prompt=outline_example_prompt,
        suffix="Title: {input}\n========\nOutline: ",
        input_variables=["input"],
        example_separator="\n\n",
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=outline_human_prompt
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chat = ChatOpenAI(temperature=0.3, max_tokens=256)
    outline_chain = LLMChain(llm=chat, prompt=chat_prompt)
    output = outline_chain.run(input=title)
    try:
        result = ast.literal_eval(output)
    except SyntaxError as err:
        print("Need to implement version of OutputFixingParser", err)
    return result
