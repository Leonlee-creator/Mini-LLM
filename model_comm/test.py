from models.local_models import flan_t5

def comm():
    text = """The Hubble Space Telescope is a large telescope in space. 
    It was launched in 1990 and is still working today. 
    Hubble has taken many amazing pictures of stars and galaxies.
    """

    summary = flan_t5(f"Summarize: {text}", max_length=50)
    print(summary)
    print("Summary:", summary[0]['generated_text'])

comm()