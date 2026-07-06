import pypdf

reader = pypdf.PdfReader('dlg_book.pdf')
text = reader.pages[141].extract_text()
print(text[:500])
