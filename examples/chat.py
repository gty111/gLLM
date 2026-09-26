import argparse

from gllm import LLM


def chat(llm: LLM):
    """Interactive REPL driving the engine's public scheduling API."""
    architecture = llm.model_runner.model_loader.architecture
    print(
        "\nWelcome to the chatbot!\n"
        "Type '\\exit' to exit the chatbot.\n"
        "Type '\\clear' to clear the chatbot's history.\n"
    )
    history = []
    while True:
        prompt = input(">>> ")
        print()
        if prompt == "\\clear":
            history = []
            continue
        elif prompt == "\\exit":
            break

        if architecture == "ChatGLMModel" and hasattr(
            llm.model_runner.tokenizer, "build_chat_input"
        ):
            tokens = (
                llm.model_runner.tokenizer.build_chat_input(
                    prompt, history=history, role="user"
                )
                .get("input_ids")
                .numpy()
                .tolist()[0]
            )
        else:
            history.append({"role": "user", "content": prompt})
            tokens = llm.model_runner.encode(history, chat=True)

        seq = llm.allocate_seq(tokens)
        llm.add_requests([seq])
        while len(llm.running_maps) != 0 or len(llm.wait_lists) != 0:
            llm.schedule(log=False)
            print(
                seq.detokenize_inc(llm.model_runner.tokenizer), end="", flush=True
            )
        print("\n")

        output_text = llm.model_runner.decode(seq[seq.raw_prompt_len :])

        if architecture == "ChatGLMModel" and hasattr(
            llm.model_runner.tokenizer, "build_chat_input"
        ):
            _, history = llm.model_runner.model.process_response(
                output_text, history
            )
        else:
            history.append({"role": "assistant", "content": output_text})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with LLM")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--master-port", type=str, default="8000")
    args = parser.parse_args()

    llm = LLM(
        args.model,
        pp_size=args.pp,
        tp_size=args.tp,
        master_port=args.master_port,
    )
    chat(llm)
