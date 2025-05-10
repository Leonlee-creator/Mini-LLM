# tag_sentences.py
def tag_and_save(input_file, output_file, tag):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            line = line.strip()
            if line:
                f.write(f"<{tag}> {line}\n")

tag_and_save("data/shona_clean.txt", "data/shona_tagged.txt", "shona")
tag_and_save("data/venda_clean.txt", "data/venda_tagged.txt", "venda")
