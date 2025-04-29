from gllm import LLM
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with LLM')
    parser.add_argument("--model",type=str,required=True)
    args = parser.parse_args()
    
    llm = LLM(args.model)
    llm.chat()
    