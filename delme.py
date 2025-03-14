from datasets import load_dataset
bhasaanuvaad = load_dataset("ai4bharat/IndicVoices-ST", "indic2en", split="nepali", streaming=True)
print(next(iter(bhasaanuvaad)))