from lexisync.chunking import chunk_pages

def test_chunk_has_page_and_file_metadata():
    pages=[{"filename":"sample.pdf","page":3,"text":"7.2 Liability. The total liability shall be limited to EUR 2 million."}]
    chunks=chunk_pages(pages)
    assert chunks
    assert chunks[0]["filename"]=="sample.pdf"
    assert chunks[0]["page"]==3
