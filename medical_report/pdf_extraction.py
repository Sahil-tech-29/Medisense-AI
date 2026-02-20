import pdfplumber 

def extract_text_from_pdf(pdf_file):
    extracted_text=""

    with pdfplumber.open(pdf_file) as pdf : 
        for pages in pdf.pages:
            page_text = pages.extract_text()
            if page_text :
                extracted_text += page_text +"\n"

    return extracted_text.strip()


def is_readable_text(text):
    # Count how many normal ASCII letters exist
    readable_chars = sum(c.isalnum() for c in text)
    total_chars = len(text)

    if total_chars == 0:
        return False

    return (readable_chars / total_chars) > 0.3
