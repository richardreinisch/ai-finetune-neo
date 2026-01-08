
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

file_to_convert = "TBD"

converter = PdfConverter(
    artifact_dict=create_model_dict()
)

rendered = converter("./data/" + file_to_convert + ".pdf")
text, _, images = text_from_rendered(rendered)

with open("./parsed/" + file_to_convert + ".md", "w") as f:
    f.write(text)

