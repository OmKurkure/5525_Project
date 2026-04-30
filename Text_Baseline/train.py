import logging
import time
import pandas as pd
import random

import chz
import tinker

from tinker import types
from tinker_cookbook import model_info, renderers, checkpoint_utils
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ACKNOWLEGEMENTS:
# We acknowlege the use of AI in writing this code. Specifically, we used AI
# modify the training setup obtained from Tinker Cookbook's supervised training
# example. We were having issues with inference and checkpointing, 
# so we used AI to help use write the inference code, normalization of predictions
# and checkpoint saving code. 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INTENT_LABELS = [
    "calendar_query",
    "calendar_set",
    "datetime_query",
    "email_query",
    "email_sendemail",
    "general_quirky",
    "news_query",
    "play_music",
    "qa_factoid",
    "weather_query",
]


@chz.chz
class Config:
    base_url: str | None = None
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_length: int = 2048
    lora_rank: int = 16
    epochs: int = 15
    train_path: str = "data/it_train_400.jsonl"
    test_path: str = "data/it_test_100.jsonl"
    log_path: str = "./checkpoints"


def create_conversation(text, intent):
    return [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. "
                "Respond with ONLY one label from this list:\n"
                + ", ".join(INTENT_LABELS)
            ),
        },
        {"role": "user", "content": f"Text: {text}"},
        {"role": "assistant", "content": intent},
    ]


def build_eval_prompt(tokenizer, text):
    prompt_str = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an intent classifier. "
        "Respond with ONLY one label from this list:\n"
        + ", ".join(INTENT_LABELS)
        + "<|eot_id|>"
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        + f"Text: {text}"
        + "<|eot_id|>"
        + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return types.ModelInput.from_ints(
        tokenizer.encode(prompt_str, add_special_tokens=False)
    )


def normalize_prediction(pred: str) -> str | None:
    pred = pred.strip().lower()

    # exact match first
    if pred in INTENT_LABELS:
        return pred

    # otherwise allow cases like:
    # "weather_query\n"
    # "The answer is weather_query"
    # "weather_query<|eot_id|>"
    for label in INTENT_LABELS:
        if label in pred:
            return label

    return None


def evaluate_accuracy(sampling_client, dataset, tokenizer, renderer):
    correct = 0

    for item in dataset:
        prompt = build_eval_prompt(tokenizer, item["text"])

        params = tinker.SamplingParams(
            max_tokens=8,
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
        )

        result = sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=1,
        ).result()

        pred = tokenizer.decode(result.sequences[0].tokens).strip().lower()
        prediction = normalize_prediction(pred)
        true_label = item["label"].strip().lower()

        print(f"Input Text: {item['text']}")
        print(f"Prediction: {prediction}")
        print("===========================================================================")

        if prediction == true_label:
            correct += 1

    return correct / len(dataset)


def main(config: Config):
    # Load JSONL
    train_df = pd.read_json(config.train_path, lines=True)
    test_df = pd.read_json(config.test_path, lines=True)

    # Keep only needed columns
    train_df = train_df[["utt", "intent"]].dropna()
    test_df = test_df[["utt", "intent"]].dropna()

    train_data = [
        {"messages": create_conversation(row["utt"], row["intent"])}
        for _, row in train_df.iterrows()
    ]

    test_data = [
        {"text": row["utt"], "label": row["intent"]}
        for _, row in test_df.iterrows()
    ]

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient(base_url=config.base_url)

    training_client = service_client.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    num_batches = len(train_data) // config.batch_size

    for epoch in range(config.epochs):
        logger.info(f"\n===== EPOCH {epoch+1} =====")

        random.shuffle(train_data)

        for batch_idx in range(num_batches):
            lr_mult = 1.0 - (batch_idx / num_batches)
            current_lr = config.learning_rate * lr_mult

            adam_params = tinker.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            batch_rows = train_data[
                batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size
            ]

            batch = [
                conversation_to_datum(
                    row["messages"],
                    renderer,
                    config.max_length,
                    renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
                for row in batch_rows
            ]

            fwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
            opt = training_client.optim_step(adam_params)

            fwd_res = fwd.result()
            _ = opt.result()

            train_logprobs = [x["logprobs"] for x in fwd_res.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            logger.info(
                f"Epoch {epoch+1} | Batch {batch_idx} | Loss {train_nll:.8f}"
            )

    checkpoint_name = "final_epoch"
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=checkpoint_name,
        log_path=config.log_path,
        kind="both",
        loop_state={"epoch": config.epochs},
        ttl_seconds=None,
    )

    logger.info(f"Saved final checkpoint: {checkpoint_name}")

    model_path = f"tinker://{training_client.model_id}/sampler_weights/{checkpoint_name}"
    sampling_client = service_client.create_sampling_client(model_path=model_path)

    acc = evaluate_accuracy(sampling_client, test_data, tokenizer, renderer)
    logger.info(f"Final Test Accuracy after {config.epochs} epochs: {acc:.4f}")

    # Write experiment results to results.txt (append mode so multiple runs accumulate)
    with open("results.txt", "a") as f:
        f.write(f"===={config.model_name}====\n")
        f.write(f"Epochs: {config.epochs}\n")
        f.write(f"learning rate: {config.learning_rate}\n")
        f.write(f"batch_size: {config.batch_size}\n")
        f.write(f"lora_rank: {config.lora_rank}\n")
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"train_file: {config.train_path}\n")
        f.write("\n")
 
    logger.info("Wrote experiment results to results.txt")


if __name__ == "__main__":
    chz.nested_entrypoint(main)