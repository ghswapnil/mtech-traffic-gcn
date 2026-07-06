import pypdf

reader = pypdf.PdfReader('dlg_book.pdf')
# Usually PDF page index is offset from printed page number. 
# Let's search around pages 110-140 for "5.2 The General GNN Frameworks"
for i in range(100, 150):
    text = reader.pages[i].extract_text()
    if "5.2 The General GNN Frameworks" in text or "5.2.1 A General Framework" in text:
        print(f"--- PAGE {i} ---")
        print(text[:1000])
