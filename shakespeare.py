

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a translator that rewrites modern English into the style of "
    "William Shakespeare's Early Modern English. Use period-appropriate "
    "vocabulary (thee, thou, thy, hath, doth, art, 'tis, prithee, forsooth), "
    "poetic phrasing, and an Elizabethan cadence. Preserve the original "
    "meaning. Output ONLY the rewritten sentence, with no explanation, no "
    "quotation marks, and no preamble."
)


def load_model():
    print(f"Loading {MODEL_ID} on {DEVICE} ... (first run downloads the model)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, 
        device_map=DEVICE,
    )
    return tokenizer, model


def to_shakespeare(text, tokenizer, model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,    
            top_p=0.9,
            repetition_penalty=1.1,
        )


    generated = output_ids[0][prompt_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result.strip()


def main():
    tokenizer, model = load_model()
    print("\nReady. Type an English sentence (or 'quit' to exit).\n")

    while True:
        try:
            text = input("English> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFarewell!")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            print("Farewell!")
            break

        shakespeare = to_shakespeare(text, tokenizer, model)
        print(f"Bard>    {shakespeare}\n")


if __name__ == "__main__":
    main()