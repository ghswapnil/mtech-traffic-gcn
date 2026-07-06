import os
try:
    import pypdf
except ImportError:
    os.system('pip install pypdf > /dev/null 2>&1')
    import pypdf

reader = pypdf.PdfReader('dlg_book.pdf')
with open('toc.txt', 'w') as f:
    for i in range(min(30, len(reader.pages))):
        text = reader.pages[i].extract_text()
        if text:
            f.write(text)
