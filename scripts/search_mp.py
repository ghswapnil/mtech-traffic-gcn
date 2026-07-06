import pypdf

reader = pypdf.PdfReader('dlg_book.pdf')
for i in range(120, 150):
    text = reader.pages[i].extract_text()
    if "message passing" in text.lower() or "message" in text.lower():
        print(f"--- PAGE {i} ---")
        lines = [line for line in text.split('\n') if "message" in line.lower()]
        for line in lines:
            print(line)
