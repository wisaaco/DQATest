from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
import re
import torch

localModel = "~/models/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(localModel,local_files_only=True)
processor.tokenizer.padding_side = 'left'
model = VisionEncoderDecoderModel.from_pretrained(localModel)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
image = dataset[1]["image"]
width, height = image.size
image.resize((int(0.3*width), (int(0.3*height))))


# move model to GPU if it's available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# prepare encoder inputs
pixel_values = processor(images=dataset['image'][:2], return_tensors="pt").pixel_values
batch_size = pixel_values.shape[0]

# prepare decoder inputs
task_prompt = "{user_input}"
questions = ["¿A qué hora se toma el café?", "In which year this report was created?"]
prompts = [task_prompt.replace("{user_input}", question) for question in questions]
decoder_input_ids = processor.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").input_ids

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

sequences = processor.batch_decode(outputs.sequences)

for seq in sequences:
  sequence = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
  sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
  print(processor.token2json(sequence))

